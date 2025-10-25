from typing import cast

import numpy as np
import pytest
import torch
import torch.distributions as dist
from gymnasium import spaces
from torch import nn

from tianshou.exploration import GaussianNoise, OUNoise
from tianshou.utils import MovAvg, RunningMeanStd
from tianshou.utils.net.common import MLP, Net
from tianshou.utils.net.continuous import RecurrentActorProb, RecurrentCritic
from tianshou.utils.torch_utils import create_uniform_action_dist, torch_train_mode


def test_noise() -> None:
    noise = GaussianNoise()
    size = (3, 4, 5)
    assert np.allclose(noise(size).shape, size)
    noise = OUNoise()
    noise.reset()
    assert np.allclose(noise(size).shape, size)


def test_moving_average() -> None:
    stat = MovAvg(10)
    assert np.allclose(stat.get(), 0)
    assert np.allclose(stat.mean(), 0)
    assert np.allclose(stat.std() ** 2, 0)
    stat.add(torch.tensor([1]))
    stat.add(np.array([2]))
    stat.add([3, 4])
    stat.add(5.0)
    assert np.allclose(stat.get(), 3)
    assert np.allclose(stat.mean(), 3)
    assert np.allclose(stat.std() ** 2, 2)


def test_rms() -> None:
    rms = RunningMeanStd()
    assert np.allclose(rms.mean, 0)
    assert np.allclose(rms.var, 1)
    rms.update(np.array([[[1, 2], [3, 5]]]))
    rms.update(np.array([[[1, 2], [3, 4]], [[1, 2], [0, 0]]]))
    assert np.allclose(rms.mean, np.array([[1, 2], [2, 3]]), atol=1e-3)
    assert np.allclose(rms.var, np.array([[0, 0], [2, 14 / 3.0]]), atol=1e-3)


def test_net() -> None:
    # here test the networks that does not appear in the other script
    bsz = 64
    # MLP
    data = torch.rand([bsz, 3])
    mlp = MLP(input_dim=3, output_dim=6, hidden_sizes=[128])
    assert list(mlp(data).shape) == [bsz, 6]
    # output == 0 and len(hidden_sizes) == 0 means identity model
    mlp = MLP(input_dim=6, output_dim=0)
    assert data.shape == mlp(data).shape
    # common net
    state_shape = (10, 2)
    action_shape = (5,)
    data = torch.rand([bsz, *state_shape])
    expect_output_shape = [bsz, *action_shape]
    net = Net(
        state_shape=state_shape,
        action_shape=action_shape,
        hidden_sizes=[128, 128],
        norm_layer=torch.nn.LayerNorm,
        activation=None,
    )
    assert list(net(data)[0].shape) == expect_output_shape
    assert str(net).count("LayerNorm") == 2
    assert str(net).count("ReLU") == 0
    Q_param = V_param = {"hidden_sizes": [128, 128]}
    net = Net(
        state_shape=state_shape,
        action_shape=action_shape,
        hidden_sizes=[128, 128],
        dueling_param=(Q_param, V_param),
    )
    assert list(net(data)[0].shape) == expect_output_shape
    # concat
    net = Net(
        state_shape=state_shape,
        action_shape=action_shape,
        hidden_sizes=[128],
        concat=True,
    )
    data = torch.rand([bsz, int(np.prod(state_shape)) + int(np.prod(action_shape))])
    expect_output_shape = [bsz, 128]
    assert list(net(data)[0].shape) == expect_output_shape
    net = Net(
        state_shape=state_shape,
        action_shape=action_shape,
        hidden_sizes=[128],
        concat=True,
        dueling_param=(Q_param, V_param),
    )
    assert list(net(data)[0].shape) == expect_output_shape
    # recurrent actor/critic
    data = torch.rand([bsz, *state_shape]).flatten(1)
    expect_output_shape = [bsz, *action_shape]
    net = RecurrentActorProb(layer_num=3, state_shape=state_shape, action_shape=action_shape)
    mu, sigma = net(data)[0]
    assert mu.shape == sigma.shape
    assert list(mu.shape) == [bsz, 5]
    net = RecurrentCritic(layer_num=3, state_shape=state_shape, action_shape=action_shape)
    data = torch.rand([bsz, 8, int(np.prod(state_shape))])
    act = torch.rand(expect_output_shape)
    assert list(net(data, act).shape) == [bsz, 1]


def test_in_eval_mode() -> None:
    module = nn.Linear(3, 4)
    module.train()
    with torch_train_mode(module, False):
        assert not module.training
    assert module.training


def test_in_train_mode() -> None:
    module = nn.Linear(3, 4)
    module.eval()
    with torch_train_mode(module):
        assert module.training
    assert not module.training


class TestCreateActionDistribution:
    @classmethod
    def setup_class(cls) -> None:
        # Set random seeds for reproducibility
        torch.manual_seed(0)
        np.random.seed(0)

    @pytest.mark.parametrize(
        "action_space, batch_size",
        [
            (spaces.Box(low=-1.0, high=1.0, shape=(3,)), 1),
            (spaces.Box(low=-1.0, high=1.0, shape=(3,)), 5),
            (spaces.Discrete(5), 1),
            (spaces.Discrete(5), 5),
        ],
    )
    def test_distribution_properties(
        self,
        action_space: spaces.Box | spaces.Discrete,
        batch_size: int,
    ) -> None:
        distribution = create_uniform_action_dist(action_space, batch_size)

        # Correct distribution type
        if isinstance(action_space, spaces.Box):
            assert isinstance(distribution, dist.Uniform)
        elif isinstance(action_space, spaces.Discrete):
            assert isinstance(distribution, dist.Categorical)

        # Samples are within correct range
        samples = distribution.sample()
        if isinstance(action_space, spaces.Box):
            low = torch.tensor(action_space.low, dtype=torch.float32)
            high = torch.tensor(action_space.high, dtype=torch.float32)
            assert torch.all(samples >= low)
            assert torch.all(samples <= high)
        elif isinstance(action_space, spaces.Discrete):
            assert torch.all(samples >= 0)
            assert torch.all(samples < action_space.n)

    @pytest.mark.parametrize(
        "action_space, batch_size",
        [
            (spaces.Box(low=-1.0, high=1.0, shape=(3,)), 1),
            (spaces.Box(low=-1.0, high=1.0, shape=(3,)), 5),
            (spaces.Discrete(5), 1),
            (spaces.Discrete(5), 5),
        ],
    )
    def test_distribution_uniformity(
        self,
        action_space: spaces.Box | spaces.Discrete,
        batch_size: int,
    ) -> None:
        distribution = create_uniform_action_dist(action_space, batch_size)

        # Test 7: Uniform distribution (statistical test)
        large_sample = distribution.sample(torch.Size((10000,)))
        if isinstance(action_space, spaces.Box):
            # For Box, check if mean is close to 0 and std is close to 1/sqrt(3)
            assert torch.allclose(large_sample.mean(), torch.tensor(0.0), atol=0.1)
            assert torch.allclose(large_sample.std(), torch.tensor(1 / 3**0.5), atol=0.1)
        elif isinstance(action_space, spaces.Discrete):
            # For Discrete, check if all actions are roughly equally likely
            n_actions = cast(int, action_space.n)
            counts = torch.bincount(large_sample.flatten(), minlength=n_actions).float()
            expected_count = 10000 * batch_size / n_actions
            assert torch.allclose(counts, torch.tensor(expected_count).float(), rtol=0.1)

    def test_unsupported_space(self) -> None:
        # Test 6: Raises ValueError for unsupported space
        with pytest.raises(ValueError):
            create_uniform_action_dist(spaces.MultiBinary(5))  # type: ignore

    @pytest.mark.parametrize(
        "space, batch_size, expected_shape, distribution_type",
        [
            (spaces.Box(low=-1.0, high=1.0, shape=(3,)), 1, (1, 3), dist.Uniform),
            (spaces.Box(low=-1.0, high=1.0, shape=(3,)), 5, (5, 3), dist.Uniform),
            (spaces.Box(low=-1.0, high=1.0, shape=(3,)), 10, (10, 3), dist.Uniform),
            (spaces.Discrete(5), 1, (1,), dist.Categorical),
            (spaces.Discrete(5), 5, (5,), dist.Categorical),
            (spaces.Discrete(5), 10, (10,), dist.Categorical),
        ],
    )
    def test_batch_sizes(
        self,
        space: spaces.Box | spaces.Discrete,
        batch_size: int,
        expected_shape: tuple[int, ...],
        distribution_type: type[dist.Distribution],
    ) -> None:
        distribution = create_uniform_action_dist(space, batch_size)

        # Check distribution type
        assert isinstance(distribution, distribution_type)

        # Check sample shape
        samples = distribution.sample()
        assert samples.shape == expected_shape

        # Check internal distribution shapes
        if isinstance(space, spaces.Box):
            distribution = cast(dist.Uniform, distribution)
            assert distribution.low.shape == expected_shape
            assert distribution.high.shape == expected_shape
        elif isinstance(space, spaces.Discrete):
            distribution = cast(dist.Categorical, distribution)
            assert distribution.probs.shape == (batch_size, space.n)
