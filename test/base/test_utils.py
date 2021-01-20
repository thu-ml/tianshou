import torch
import numpy as np

from tianshou.utils import MovAvg
from tianshou.utils import SummaryWriter
from tianshou.utils.net.common import MLP, Net
from tianshou.exploration import GaussianNoise, OUNoise
from tianshou.utils.net.continuous import RecurrentActorProb, RecurrentCritic


def test_noise():
    noise = GaussianNoise()
    size = (3, 4, 5)
    assert np.allclose(noise(size).shape, size)
    noise = OUNoise()
    noise.reset()
    assert np.allclose(noise(size).shape, size)


def test_moving_average():
    stat = MovAvg(10)
    assert np.allclose(stat.get(), 0)
    assert np.allclose(stat.mean(), 0)
    assert np.allclose(stat.std() ** 2, 0)
    stat.add(torch.tensor([1]))
    stat.add(np.array([2]))
    stat.add([3, 4])
    stat.add(5.)
    assert np.allclose(stat.get(), 3)
    assert np.allclose(stat.mean(), 3)
    assert np.allclose(stat.std() ** 2, 2)


def test_net():
    # here test the networks that does not appear in the other script
    bsz = 64
    # MLP
    data = torch.rand([bsz, 3])
    mlp = MLP(3, 6, hidden_sizes=[128])
    assert list(mlp(data).shape) == [bsz, 6]
    # output == 0 and len(hidden_sizes) == 0 means identity model
    mlp = MLP(6, 0)
    assert data.shape == mlp(data).shape
    # common net
    state_shape = (10, 2)
    action_shape = (5, )
    data = torch.rand([bsz, *state_shape])
    expect_output_shape = [bsz, *action_shape]
    net = Net(state_shape, action_shape, hidden_sizes=[128, 128],
              norm_layer=torch.nn.LayerNorm, activation=None)
    assert list(net(data)[0].shape) == expect_output_shape
    assert str(net).count("LayerNorm") == 2
    assert str(net).count("ReLU") == 0
    Q_param = V_param = {"hidden_sizes": [128, 128]}
    net = Net(state_shape, action_shape, hidden_sizes=[128, 128],
              dueling_param=(Q_param, V_param))
    assert list(net(data)[0].shape) == expect_output_shape
    # concat
    net = Net(state_shape, action_shape, hidden_sizes=[128],
              concat=True)
    data = torch.rand([bsz, np.prod(state_shape) + np.prod(action_shape)])
    expect_output_shape = [bsz, 128]
    assert list(net(data)[0].shape) == expect_output_shape
    net = Net(state_shape, action_shape, hidden_sizes=[128],
              concat=True, dueling_param=(Q_param, V_param))
    assert list(net(data)[0].shape) == expect_output_shape
    # recurrent actor/critic
    data = torch.rand([bsz, *state_shape]).flatten(1)
    expect_output_shape = [bsz, *action_shape]
    net = RecurrentActorProb(3, state_shape, action_shape)
    mu, sigma = net(data)[0]
    assert mu.shape == sigma.shape
    assert list(mu.shape) == [bsz, 5]
    net = RecurrentCritic(3, state_shape, action_shape)
    data = torch.rand([bsz, 8, np.prod(state_shape)])
    act = torch.rand(expect_output_shape)
    assert list(net(data, act).shape) == [bsz, 1]


def test_summary_writer():
    # get first instance by key of `default` or your own key
    writer1 = SummaryWriter.get_instance(
        key="first", log_dir="log/test_sw/first")
    assert writer1.log_dir == "log/test_sw/first"
    writer2 = SummaryWriter.get_instance()
    assert writer1 is writer2
    # create new instance by specify a new key
    writer3 = SummaryWriter.get_instance(
        key="second", log_dir="log/test_sw/second")
    assert writer3.log_dir == "log/test_sw/second"
    writer4 = SummaryWriter.get_instance(key="second")
    assert writer3 is writer4
    assert writer1 is not writer3
    assert writer1.log_dir != writer4.log_dir


if __name__ == '__main__':
    test_noise()
    test_moving_average()
    test_net()
    test_summary_writer()
