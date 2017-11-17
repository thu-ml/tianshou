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

#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED

#include "config.h"
#include <vector>
#include <string>
#include <bitset>
#include <memory>
#include <array>

#ifdef USE_OPENCL
#include <atomic>
class UCTNode;
#endif

#include "FastState.h"
#include "GameState.h"

class Network {
public:
    enum Ensemble {
        DIRECT, RANDOM_ROTATION
    };
    static const int board_size = 19;
    using BoardPlane = std::bitset<board_size * board_size>;
    using NNPlanes = std::vector<BoardPlane>;
    using scored_node = std::pair<float, int>;
    using Netresult = std::pair<std::vector<scored_node>, float>;

    static Netresult get_scored_moves(GameState * state,
                                      Ensemble ensemble,
                                      int rotation = -1);
    static constexpr int INPUT_CHANNELS = 18;
    static constexpr int MAX_CHANNELS = 256;

    static void initialize();
    static void benchmark(GameState * state);
    static void show_heatmap(FastState * state, Netresult & netres, bool topmoves);
    static void softmax(const std::vector<float>& input,
                        std::vector<float>& output,
                        float temperature = 1.0f);
    // tianshou_code
    static void show_once(std::string hash_key) {
        printf("%s\n", hash_key.c_str());
    }

private:
    static Netresult get_scored_moves_internal(
      GameState * state, NNPlanes & planes, int rotation);
    static void gather_features(GameState * state, NNPlanes & planes);
    static int rotate_nn_idx(const int vertex, int symmetry);
    static int rev_rotate_nn_idx(const int vertex, int symmetry);
};

#endif
