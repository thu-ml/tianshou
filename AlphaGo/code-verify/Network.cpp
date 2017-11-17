/*
    This file is part of Leela Zero.
    Copyright (C) 2017 Gian-Carlo Pascutto

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "config.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <memory>
#include <cmath>
#include <array>
#include <thread>
#include <boost/utility.hpp>
#include <boost/format.hpp>
//#include <unistd.h>
#include <string.h>

#include "Im2Col.h"

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif
#ifdef USE_MKL
#include <mkl.h>
#endif
#ifdef USE_OPENBLAS

#include <cblas.h>

#endif
#ifdef USE_OPENCL

#include "OpenCL.h"
#include "UCTNode.h"

#endif

#include "SGFTree.h"
#include "SGFParser.h"
#include "Utils.h"
#include "FastBoard.h"
#include "Random.h"
#include "Network.h"
#include "GTP.h"
#include "Utils.h"

using namespace Utils;

// Input + residual block tower
std::vector<std::vector<float>> conv_weights;
std::vector<std::vector<float>> conv_biases;
std::vector<std::vector<float>> batchnorm_means;
std::vector<std::vector<float>> batchnorm_variances;

// Policy head
std::vector<float> conv_pol_w;
std::vector<float> conv_pol_b;
std::array<float, 2> bn_pol_w1;
std::array<float, 2> bn_pol_w2;

std::array<float, 261364> ip_pol_w;
std::array<float, 362> ip_pol_b;

// Value head
std::vector<float> conv_val_w;
std::vector<float> conv_val_b;
std::array<float, 1> bn_val_w1;
std::array<float, 1> bn_val_w2;

std::array<float, 92416> ip1_val_w;
std::array<float, 256> ip1_val_b;

std::array<float, 256> ip2_val_w;
std::array<float, 1> ip2_val_b;

void Network::benchmark(GameState *state) {
    {
        int BENCH_AMOUNT = 1600;
        int cpus = cfg_num_threads;
        int iters_per_thread = (BENCH_AMOUNT + (cpus - 1)) / cpus;

        Time start;

        ThreadGroup tg(thread_pool);
        for (int i = 0; i < cpus; i++) {
            tg.add_task([iters_per_thread, state]() {
                GameState mystate = *state;
                for (int loop = 0; loop < iters_per_thread; loop++) {
                    auto vec = get_scored_moves(&mystate, Ensemble::RANDOM_ROTATION);
                }
            });
        };
        tg.wait_all();

        Time end;

        myprintf("%5d evaluations in %5.2f seconds -> %d n/s\n",
                 BENCH_AMOUNT,
                 (float) Time::timediff(start, end) / 100.0,
                 (int) ((float) BENCH_AMOUNT / ((float) Time::timediff(start, end) / 100.0)));
    }
}

void Network::initialize(void) {
#ifdef USE_OPENCL
    myprintf("Initializing OpenCL\n");
    opencl.initialize();

    // Count size of the network
    myprintf("Detecting residual layers...");
    std::ifstream wtfile(cfg_weightsfile);
    if (wtfile.fail()) {
        myprintf("Could not open weights file: %s\n", cfg_weightsfile.c_str());
        exit(EXIT_FAILURE);
    }
    std::string line;
    auto linecount = size_t{0};
    while (std::getline(wtfile, line)) {
        // Second line of parameters are the convolution layer biases,
        // so this tells us the amount of channels in the residual layers.
        // (Provided they're all equally large - that's not actually required!)
        if (linecount == 1) {
            std::stringstream ss(line);
            auto count = std::distance(std::istream_iterator<std::string>(ss),
                                       std::istream_iterator<std::string>());
            myprintf("%d channels...", count);
        }
        linecount++;
    }
    // 1 input layer (4 x weights), 14 ending weights, the rest are residuals
    // every residual has 8 x weight lines
    auto residual_layers = linecount - (4 + 14);
    if (residual_layers % 8 != 0) {
        myprintf("\nInconsistent number of weights in the file.\n");
        exit(EXIT_FAILURE);
    }
    residual_layers /= 8;
    myprintf("%d layers\nTransferring weights to GPU...", residual_layers);

    // Re-read file and process
    wtfile.clear();
    wtfile.seekg(0, std::ios::beg);

    auto plain_conv_layers = 1 + (residual_layers * 2);
    auto plain_conv_wts = plain_conv_layers * 4;
    linecount = 0;
    while (std::getline(wtfile, line)) {
        std::vector<float> weights;
        float weight;
        std::istringstream iss(line);
        while (iss >> weight) {
            weights.emplace_back(weight);
        }
        if (linecount < plain_conv_wts) {
            if (linecount % 4 == 0) {
                conv_weights.emplace_back(weights);
            } else if (linecount % 4 == 1) {
                conv_biases.emplace_back(weights);
            } else if (linecount % 4 == 2) {
                batchnorm_means.emplace_back(weights);
            } else if (linecount % 4 == 3) {
                batchnorm_variances.emplace_back(weights);
            }
        } else if (linecount == plain_conv_wts) {
            conv_pol_w = std::move(weights);
        } else if (linecount == plain_conv_wts + 1) {
            conv_pol_b = std::move(weights);
        } else if (linecount == plain_conv_wts + 2) {
            std::copy(begin(weights), end(weights), begin(bn_pol_w1));
        } else if (linecount == plain_conv_wts + 3) {
            std::copy(begin(weights), end(weights), begin(bn_pol_w2));
        } else if (linecount == plain_conv_wts + 4) {
            std::copy(begin(weights), end(weights), begin(ip_pol_w));
        } else if (linecount == plain_conv_wts + 5) {
            std::copy(begin(weights), end(weights), begin(ip_pol_b));
        } else if (linecount == plain_conv_wts + 6) {
            conv_val_w = std::move(weights);
        } else if (linecount == plain_conv_wts + 7) {
            conv_val_b = std::move(weights);
        } else if (linecount == plain_conv_wts + 8) {
            std::copy(begin(weights), end(weights), begin(bn_val_w1));
        } else if (linecount == plain_conv_wts + 9) {
            std::copy(begin(weights), end(weights), begin(bn_val_w2));
        } else if (linecount == plain_conv_wts + 10) {
            std::copy(begin(weights), end(weights), begin(ip1_val_w));
        } else if (linecount == plain_conv_wts + 11) {
            std::copy(begin(weights), end(weights), begin(ip1_val_b));
        } else if (linecount == plain_conv_wts + 12) {
            std::copy(begin(weights), end(weights), begin(ip2_val_w));
        } else if (linecount == plain_conv_wts + 13) {
            std::copy(begin(weights), end(weights), begin(ip2_val_b));
        }
        linecount++;
    }
    wtfile.close();

    // input
    size_t weight_index = 0;
    opencl_net.push_convolve(3, conv_weights[weight_index],
                             conv_biases[weight_index]);
    opencl_net.push_batchnorm(361, batchnorm_means[weight_index],
                              batchnorm_variances[weight_index]);
    weight_index++;

    // residual blocks
    for (auto i = size_t{0}; i < residual_layers; i++) {
        opencl_net.push_residual(3, conv_weights[weight_index],
                                 conv_biases[weight_index],
                                 batchnorm_means[weight_index],
                                 batchnorm_variances[weight_index],
                                 conv_weights[weight_index + 1],
                                 conv_biases[weight_index + 1],
                                 batchnorm_means[weight_index + 1],
                                 batchnorm_variances[weight_index + 1]);
        weight_index += 2;
    }
    myprintf("done\n");
#endif
#ifdef USE_BLAS
#ifndef __APPLE__
#ifdef USE_OPENBLAS
    openblas_set_num_threads(1);
    myprintf("BLAS Core: %s\n", openblas_get_corename());
#endif
#ifdef USE_MKL
    //mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL);
    mkl_set_num_threads(1);
    MKLVersion Version;
    mkl_get_version(&Version);
    myprintf("BLAS core: MKL %s\n", Version.Processor);
#endif
#endif
#endif
}

#ifdef USE_BLAS

template<unsigned int filter_size,
        unsigned int outputs>
void convolve(const std::vector<float> &input,
              const std::vector<float> &weights,
              const std::vector<float> &biases,
              std::vector<float> &output) {
    // fixed for 19x19
    constexpr unsigned int width = 19;
    constexpr unsigned int height = 19;
    constexpr unsigned int spatial_out = width * height;
    constexpr unsigned int filter_len = filter_size * filter_size;

    auto channels = int(weights.size() / (biases.size() * filter_len));
    unsigned int filter_dim = filter_len * channels;

    std::vector<float> col(filter_dim * width * height);
    im2col<filter_size>(channels, input, col);

    // Weight shape (output, input, filter_size, filter_size)
    // 96 22 5 5
    // outputs[96,19x19] = weights[96,22x9] x col[22x9,19x19]
    // C←αAB + βC
    // M Number of rows in matrices A and C.
    // N Number of columns in matrices B and C.
    // K Number of columns in matrix A; number of rows in matrix B.
    // lda The size of the first dimention of matrix A; if you are
    // passing a matrix A[m][n], the value should be m.
    //    cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
    //                ldb, beta, C, N);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            // M        N            K
                outputs, spatial_out, filter_dim,
                1.0f, &weights[0], filter_dim,
                &col[0], spatial_out,
                0.0f, &output[0], spatial_out);

    for (unsigned int o = 0; o < outputs; o++) {
        for (unsigned int b = 0; b < spatial_out; b++) {
            output[(o * spatial_out) + b] =
                    biases[o] + output[(o * spatial_out) + b];
        }
    }
}

template<unsigned int inputs,
        unsigned int outputs,
        size_t W, size_t B>
void innerproduct(const std::vector<float> &input,
                  const std::array<float, W> &weights,
                  const std::array<float, B> &biases,
                  std::vector<float> &output) {
    assert(B == outputs);

    cblas_sgemv(CblasRowMajor, CblasNoTrans,
            // M     K
                outputs, inputs,
                1.0f, &weights[0], inputs,
                &input[0], 1,
                0.0f, &output[0], 1);

    auto lambda_ReLU = [](float val) {
        return (val > 0.0f) ?
               val : 0.0f;
    };

    for (unsigned int o = 0; o < outputs; o++) {
        float val = biases[o] + output[o];
        if (outputs == 256) {
            val = lambda_ReLU(val);
        }
        output[o] = val;
    }
}

template<unsigned int channels,
        unsigned int spatial_size>
void batchnorm(const std::vector<float> &input,
               const std::array<float, channels> &means,
               const std::array<float, channels> &variances,
               std::vector<float> &output) {
    constexpr float epsilon = 1e-5f;

    auto lambda_ReLU = [](float val) {
        return (val > 0.0f) ?
               val : 0.0f;
    };

    for (unsigned int c = 0; c < channels; ++c) {
        float mean = means[c];
        float variance = variances[c] + epsilon;
        float scale_stddiv = 1.0f / std::sqrt(variance);

        float *out = &output[c * spatial_size];
        float const *in = &input[c * spatial_size];
        for (unsigned int b = 0; b < spatial_size; b++) {
            out[b] = lambda_ReLU(scale_stddiv * (in[b] - mean));
        }
    }
}

#endif

void Network::softmax(const std::vector<float> &input,
                      std::vector<float> &output,
                      float temperature) {
    assert(&input != &output);

    float alpha = *std::max_element(input.begin(),
                                    input.begin() + output.size());
    alpha /= temperature;

    float denom = 0.0f;
    std::vector<float> helper(output.size());
    for (size_t i = 0; i < output.size(); i++) {
        float val = std::exp((input[i] / temperature) - alpha);
        helper[i] = val;
        denom += val;
    }
    for (size_t i = 0; i < output.size(); i++) {
        output[i] = helper[i] / denom;
    }
}

/* magic code, only execute once, what is the mechanism?
void function() {
    static const auto runOnce = [] (auto content) { std::cout << content << std::endl; return true;};
}
 */

std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
    if (!pipe) throw std::runtime_error("popen() failed!");
    while (!feof(pipe.get())) {
        if (fgets(buffer.data(), 128, pipe.get()) != nullptr)
            result += buffer.data();
    }
    return result;
}

Network::Netresult Network::get_scored_moves(
        GameState *state, Ensemble ensemble, int rotation) {
    Netresult result;
    if (state->board.get_boardsize() != 19) {
        return result;
    }

    NNPlanes planes;
    gather_features(state, planes);

    /*
     * tianshou_code
     * writing the board information to local file
     */
    //static std::once_flag plane_size_flag;
    //std::call_once ( plane_size_flag, [&]{ printf("Network::get_scored_moves planses size : %d\n", planes.size());} );
    //std::call_once ( plane_content_flag, [&]{
    /*
    for (int i = 0; i < board_size; ++i) {
        for (int j = 0; j < board_size; ++j) {
            std::cout << planes[k][i * board_size + j] << " ";
        }
        printf("\n");
    }
    printf("======================================\n");
     */
    // } );
    static int call_number = 0;
    call_number++;
    std::ofstream mctsnn_file;
    mctsnn_file.open("/home/yama/rl/tianshou/leela-zero/src/mcts_nn_files/board_" + std::to_string(call_number));
    const int total_count = board_size * board_size;
    for (int k = 0; k < planes.size(); ++k) {
        for (int i = 0; i < total_count; ++i) {
            mctsnn_file << planes[k][i];
        }
        mctsnn_file << '\n';
    }
    mctsnn_file.close();

    const int extended_board_size = board_size + 2;
    Network::Netresult nn_res;
    std::string cmd = "python /home/yama/rl/tianshou/AlphaGo/Network.py " + std::to_string(call_number);
    std::string res = exec(cmd.c_str());
    //std::cout << res << std::endl;
    std::string buf; // Have a buffer string
    std::stringstream ss(res); // Insert the string into a stream
    const int policy_size = board_size * board_size + 1;
    int idx = 0;
    for (int i = 0; i < extended_board_size; ++i) {
        for (int j = 0; j < extended_board_size; ++j) {
            if ((0 < i) && (i < board_size + 1) && (0 < j) && (j < board_size + 1)) {
                ss >> buf;
                nn_res.first.emplace_back(std::make_pair(std::stod(buf), idx));
                //std::cout << std::fixed << "[" << std::stod(buf) << "," << idx << "]\n";
            }
            idx++;
        }
    }
    // probability of pass
    ss >> buf;
    nn_res.first.emplace_back(std::make_pair(std::stod(buf), -1));
    std::cout << "tianshou nn output : \t\n";
    auto max_iterator = std::max_element(nn_res.first.begin(), nn_res.first.end());
    int argmax = std::distance(nn_res.first.begin(), max_iterator);
    int line = (argmax / board_size) + 1, column = (argmax % board_size) + 1;
    std::cout << "\tmove  : " << argmax << " [" << line << "," << column << "]" << std::endl;
    // evaluation of state value
    ss >> buf;
    nn_res.second = std::stod(buf);
    std::cout << "\tvalue : " << nn_res.second << std::endl;

    if (ensemble == DIRECT) {
        assert(rotation >= 0 && rotation <= 7);
        result = get_scored_moves_internal(state, planes, rotation);
    } else {
        assert(ensemble == RANDOM_ROTATION);
        assert(rotation == -1);
        int rand_rot = Random::get_Rng()->randfix<8>();
        std::cout << "rotation : " << rand_rot << std::endl;
        result = get_scored_moves_internal(state, planes, rand_rot);
    }

    /*
    static std::once_flag of;
    std::call_once(of, [&] { } );
    for (auto ele: result.first) {
        std::cout << std::fixed << "[" << ele.first << "," << ele.second << "]\n";
    }
     */
    std::cout << "leela nn output : \t\n";
    max_iterator = std::max_element(result.first.begin(), result.first.end());

    argmax = std::distance(result.first.begin(), max_iterator);
    line = (argmax / board_size) + 1, column = (argmax % board_size) + 1;
    std::cout << "\tmove  : " << argmax << " [" << line << "," << column << "]" << std::endl;
    std::cout << "\tvalue : " << result.second << std::endl;

    return nn_res;
    //return result;
}

Network::Netresult Network::get_scored_moves_internal(
        GameState *state, NNPlanes &planes, int rotation) {
    assert(rotation >= 0 && rotation <= 7);
    constexpr int channels = INPUT_CHANNELS;
    assert(channels == planes.size());
    constexpr int width = 19;
    constexpr int height = 19;
    constexpr int max_channels = MAX_CHANNELS;
    std::vector<float> input_data(max_channels * width * height);
    std::vector<float> output_data(max_channels * width * height);
    std::vector<float> policy_data_1(2 * width * height);
    std::vector<float> policy_data_2(2 * width * height);
    std::vector<float> value_data_1(1 * width * height);
    std::vector<float> value_data_2(1 * width * height);
    std::vector<float> policy_out((width * height) + 1);
    std::vector<float> softmax_data((width * height) + 1);
    std::vector<float> winrate_data(256);
    std::vector<float> winrate_out(1);
    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int vtx = rotate_nn_idx(h * 19 + w, rotation);
                input_data[(c * height + h) * width + w] =
                        (float) planes[c][vtx];
            }
        }
    }
#ifdef USE_OPENCL
    opencl_net.forward(input_data, output_data);
    // Get the moves
    convolve<1, 2>(output_data, conv_pol_w, conv_pol_b, policy_data_1);
    batchnorm<2, 361>(policy_data_1, bn_pol_w1, bn_pol_w2, policy_data_2);
    innerproduct<2 * 361, 362>(policy_data_2, ip_pol_w, ip_pol_b, policy_out);
    softmax(policy_out, softmax_data, cfg_softmax_temp);
    std::vector<float> &outputs = softmax_data;

    // Now get the score
    convolve<1, 1>(output_data, conv_val_w, conv_val_b, value_data_1);
    batchnorm<1, 361>(value_data_1, bn_val_w1, bn_val_w2, value_data_2);
    innerproduct<361, 256>(value_data_2, ip1_val_w, ip1_val_b, winrate_data);
    innerproduct<256, 1>(winrate_data, ip2_val_w, ip2_val_b, winrate_out);

    // Sigmoid
    float winrate_sig = (1.0f + std::tanh(winrate_out[0])) / 2.0f;
#elif defined(USE_BLAS) && !defined(USE_OPENCL)
#error "Not implemented"
    // Not implemented yet - not very useful unless you have some
    // sort of Xeon Phi
    softmax(output_data, softmax_data, cfg_softmax_temp);
    // Move scores
    std::vector<float>& outputs = softmax_data;
#endif
    std::vector<scored_node> result;
    for (size_t idx = 0; idx < outputs.size(); idx++) {
        if (idx < 19 * 19) {
            auto rot_idx = rev_rotate_nn_idx(idx, rotation);
            auto val = outputs[rot_idx];
            int x = idx % 19;
            int y = idx / 19;
            int vtx = state->board.get_vertex(x, y);
            if (state->board.get_square(vtx) == FastBoard::EMPTY) {
                result.emplace_back(val, vtx);
            }
        } else {
            result.emplace_back(outputs[idx], FastBoard::PASS);
        }
    }

    return std::make_pair(result, winrate_sig);
}

void Network::show_heatmap(FastState *state, Netresult &result, bool topmoves) {
    auto moves = result.first;
    std::vector<std::string> display_map;
    std::string line;

    for (unsigned int y = 0; y < 19; y++) {
        for (unsigned int x = 0; x < 19; x++) {
            int vtx = state->board.get_vertex(x, y);

            auto item = std::find_if(moves.cbegin(), moves.cend(),
                                     [&vtx](scored_node const &item) {
                                         return item.second == vtx;
                                     });

            float score = 0.0f;
            // Non-empty squares won't be scored
            if (item != moves.end()) {
                score = item->first;
                assert(vtx == item->second);
            }

            line += boost::str(boost::format("%3d ") % int(score * 1000));
            if (x == 18) {
                display_map.push_back(line);
                line.clear();
            }
        }
    }

    for (int i = display_map.size() - 1; i >= 0; --i) {
        myprintf("%s\n", display_map[i].c_str());
    }
    assert(result.first.back().second == FastBoard::PASS);
    int pass_score = int(result.first.back().first * 1000);
    myprintf("pass: %d\n", pass_score);
    myprintf("winrate: %f\n", result.second);

    if (topmoves) {
        std::stable_sort(moves.rbegin(), moves.rend());

        float cum = 0.0f;
        size_t tried = 0;
        while (cum < 0.85f && tried < moves.size()) {
            if (moves[tried].first < 0.01f) break;
            myprintf("%1.3f (%s)\n",
                     moves[tried].first,
                     state->board.move_to_text(moves[tried].second).c_str());
            cum += moves[tried].first;
            tried++;
        }
    }
}

void Network::gather_features(GameState *state, NNPlanes &planes) {
    planes.resize(18);
    const size_t our_offset = 0;
    const size_t their_offset = 8;
    BoardPlane &black_to_move = planes[16];
    BoardPlane &white_to_move = planes[17];

    bool whites_move = state->get_to_move() == FastBoard::WHITE;
    // tianshou_code
    //std::cout << "whites_move : " << whites_move << std::endl;
    if (whites_move) {
        white_to_move.set();
    } else {
        black_to_move.set();
    }

    // Go back in time, fill history boards
    size_t backtracks = 0;
    for (int h = 0; h < 8; h++) {
        int tomove = state->get_to_move();
        // collect white, black occupation planes
        for (int j = 0; j < 19; j++) {
            for (int i = 0; i < 19; i++) {
                int vtx = state->board.get_vertex(i, j);
                FastBoard::square_t color =
                        state->board.get_square(vtx);
                int idx = j * 19 + i;
                if (color != FastBoard::EMPTY) {
                    if (color == tomove) {
                        planes[our_offset + h][idx] = true;
                    } else {
                        planes[their_offset + h][idx] = true;
                    }
                }
            }
        }
        if (!state->undo_move()) {
            break;
        } else {
            backtracks++;
        }
    }

    // Now go back to present day
    for (size_t h = 0; h < backtracks; h++) {
        state->forward_move();
    }
}

int Network::rev_rotate_nn_idx(const int vertex, int symmetry) {
    static const int invert[] = {0, 1, 2, 3, 4, 6, 5, 7};
    assert(rotate_nn_idx(rotate_nn_idx(vertex, symmetry), invert[symmetry])
           == vertex);
    return rotate_nn_idx(vertex, invert[symmetry]);
}

int Network::rotate_nn_idx(const int vertex, int symmetry) {
    assert(vertex >= 0 && vertex < 19 * 19);
    assert(symmetry >= 0 && symmetry < 8);
    int x = vertex % 19;
    int y = vertex / 19;
    int newx;
    int newy;

    if (symmetry >= 4) {
        std::swap(x, y);
        symmetry -= 4;
    }

    if (symmetry == 0) {
        newx = x;
        newy = y;
    } else if (symmetry == 1) {
        newx = x;
        newy = 19 - y - 1;
    } else if (symmetry == 2) {
        newx = 19 - x - 1;
        newy = y;
    } else {
        assert(symmetry == 3);
        newx = 19 - x - 1;
        newy = 19 - y - 1;
    }

    int newvtx = (newy * 19) + newx;
    assert(newvtx >= 0 && newvtx < 19 * 19);
    return newvtx;
}
