import torch
import numpy as np

from tianshou.utils import MovAvg
from tianshou.utils import SummaryWriter
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import DQN
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
    # common net
    state_shape = (10, 2)
    action_shape = (5, )
    data = torch.rand([bsz, *state_shape])
    expect_output_shape = [bsz, *action_shape]
    net = Net(3, state_shape, action_shape, norm_layer=torch.nn.LayerNorm)
    assert list(net(data)[0].shape) == expect_output_shape
    net = Net(3, state_shape, action_shape, dueling=(2, 2))
    assert list(net(data)[0].shape) == expect_output_shape
    # recurrent actor/critic
    data = data.flatten(1)
    net = RecurrentActorProb(3, state_shape, action_shape)
    mu, sigma = net(data)[0]
    assert mu.shape == sigma.shape
    assert list(mu.shape) == [bsz, 5]
    net = RecurrentCritic(3, state_shape, action_shape)
    data = torch.rand([bsz, 8, np.prod(state_shape)])
    act = torch.rand(expect_output_shape)
    assert list(net(data, act).shape) == [bsz, 1]
    # DQN
    state_shape = (4, 84, 84)
    action_shape = (6, )
    data = np.random.rand(bsz, *state_shape)
    expect_output_shape = [bsz, *action_shape]
    net = DQN(*state_shape, action_shape)
    assert list(net(data)[0].shape) == expect_output_shape


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
