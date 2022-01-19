import itertools
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.distributions import Independent, Normal
from torch.utils.data import DataLoader

from tianshou.data import Batch, ReplayBuffer
from tianshou.data.dataset import TransitionDataset, TransitionEnsembleDataset
from tianshou.utils.net.common import GaussianMLELoss


class GaussianModel(object):
    """Wrapper of Gaussian model.

    :param int ensemble_size: number of subnets in the ensemble.
    :param torch.nn.Module network: core network of learned model
    :param torch.nn.Optimizer optimizer: network optimizer
    :param Optional[Union[str, int, torch.device]] device:
    :param int env_num: number of environments to be executed in parallel.
    :param int batch_size: model training batch size.
    :param float ratio: train-validation split ratio.
    :param bool deterministic: whether to predict the next observation
        deterministically.
    :param Optional[int] max_epoch: maximum number of epochs of each training.
    :param int max_static_epoch: If validation error is not reduced by a certain
        threshold for max_static_epoch epochs, the training is early-stopped.
    """

    def __init__(
        self,
        ensemble_size: int,
        network: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: Optional[Union[str, int, torch.device]] = None,
        num_elites: int = 1,
        batch_size: int = 64,
        ratio: float = 0.8,
        deterministic: bool = False,
        max_epoch: Optional[int] = None,
        max_static_epoch: int = 5,
    ) -> None:
        self.ensemble_size = ensemble_size
        self.network = network
        self.optimizer = optimizer
        self.device = device
        self.num_elites = num_elites
        self.batch_size = batch_size
        self.ratio = ratio
        self.deterministic = deterministic
        self.max_epoch = max_epoch
        self.max_static_epoch = max_static_epoch
        self.best = [1e10] * ensemble_size

    def train(
        self,
        batch: Batch,
    ) -> Dict[str, Union[float, int]]:
        """Train the dynamics model.

        :param tianshou.data.Batch batch: Training data

        :return: Training information including training loss, validation loss and
        number of training epochs.
        """
        batch.to_torch(dtype=torch.float32)
        total_num = batch.obs.shape[0]
        train_num = int(total_num * self.ratio)
        permutation = np.random.permutation(total_num)
        train_dataset = TransitionEnsembleDataset(
            batch=batch[permutation[:train_num]],
            ensemble_size=self.ensemble_size,
        )
        val_dataset = TransitionDataset(batch=batch[permutation[train_num:]], )
        train_dl = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_dl = DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
        )

        epoch_iter: Iterable
        if self.max_epoch is None:
            epoch_iter = itertools.count()
        else:
            epoch_iter = range(self.max_epoch)

        loss_fn = GaussianMLELoss(coeff=0.01)
        epochs_since_update = 0
        epoch_this_train = 0
        for _ in epoch_iter:
            self.network.train()
            for x, y in train_dl:
                # Input shape is (batch_size, ensemble_size, data_dimension)
                x = x.transpose(0, 1).to(self.device)
                y = y.transpose(0, 1).to(self.device)
                mean, logvar, max_logvar, min_logvar = self.network(x)
                loss = loss_fn(mean, logvar, max_logvar, min_logvar, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.network.eval()
            mse = torch.zeros(self.ensemble_size)
            with torch.no_grad():
                for i, (x, y) in enumerate(val_dl):
                    x = x.to(self.device)
                    y = y.to(self.device)
                    mean, _, _, _ = self.network(x)
                    batch_mse = torch.mean(torch.square(mean - y), dim=(1, 2)).cpu()
                    mse = (mse * i + batch_mse) / (i + 1)

            epoch_this_train += 1
            updated = False
            for i in range(len(mse)):
                mse_item = mse[i].item()
                improvement = (self.best[i] - mse_item) / self.best[i]
                if improvement > 0.01:
                    updated = True
                    self.best[i] = mse_item
            if updated:
                epochs_since_update = 0
            else:
                epochs_since_update += 1

            if epochs_since_update > self.max_static_epoch:
                break

        # Select elites
        self.elite_indice = torch.argsort(mse)[:self.num_elites]
        elite_mse = mse[self.elite_indice].mean().item()

        # Collect training info to be logged.
        train_info = {
            "model/train_loss": loss.item(),
            "model/val_loss": elite_mse,
            "model/train_epoch": epoch_this_train,
        }

        return train_info

    def predict(
        self,
        batch: Batch,
    ) -> Batch:
        """Predict a step forward.

        :param tianshou.data.Batch batch: prediction input

        :return: The input batch with next observation, reward and info added.
        """
        batch.to_torch(dtype=torch.float32, device=self.device)
        self.network.eval()
        observation = batch.obs
        action = batch.act
        inputs = torch.cat((observation, action), dim=-1)
        with torch.no_grad():
            mean, logvar, _, _ = self.network(inputs)
            std = torch.sqrt(torch.exp(logvar))
            dist = Independent(Normal(mean, std), 1)
            if self.deterministic:
                sample = mean
            else:
                sample = dist.rsample()
            log_prob = dist.log_prob(sample)
            # For each input, choose a network from the ensemble
            _, batch_size, _ = sample.shape
            indice = torch.randint(self.num_elites, size=(batch_size, ))
            choice_indice = self.elite_indice[indice]
            batch_indice = torch.arange(batch_size)
            next_observation = observation + \
                sample[choice_indice, batch_indice, :-1]
            reward = sample[choice_indice, batch_indice, -1]
            log_prob = log_prob[choice_indice, batch_indice]
            info = list(
                map(lambda x: {"log_prob": x.item()}, torch.split(log_prob, 1))
            )

            batch.obs_next = next_observation
            batch.rew = reward
            batch.info = info

        return batch


class FakeEnv(object):
    """Virtual environment with learned model.

    :param model: transition model.
    :param buffer: environment buffer to sample the initial observations.
    :param function terminal_fn: terminal function
    :param int env_num: Number of environments to be executed in parallel.
    """

    def __init__(
        self,
        model: GaussianModel,
        buffer: ReplayBuffer,
        terminal_fn: Callable,
        env_num: int = 1,
    ) -> None:
        self.model = model
        self.buffer = buffer
        self.env_num = env_num
        self.terminal_fn = terminal_fn

        # To be compatible with Collector
        self.action_space = None

    def __len__(self) -> int:
        return self.env_num

    def reset(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[np.ndarray, None]:
        """Reset the virtual environments.

        Sampling observations from the buffer.
        """
        if len(self.buffer) == 0:
            return None

        batch, _ = self.buffer.sample(batch_size=self.env_num, )
        self.state = batch.obs.copy()

        return self.state.copy()

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List]:
        """Take a step in every virtual environment.

        If an environment is terminated, it is automatically reset.

        :param np.ndarray action: Actions of shape (batch_size, action_dim)

        :return: Vectorized results, similar to that of OpenAI Gym environments.
        """
        observation = self.state.copy()
        batch = Batch(
            obs=observation,
            act=action,
        )
        batch: Batch = self.model.predict(batch)
        batch.to_numpy()
        reward = batch.rew
        next_observation = batch.obs_next
        done: np.ndarray = self.terminal_fn(observation, action, next_observation)

        # Reset terminal environments.
        if np.any(done):
            done_indices = np.where(done)[0]
            batch_sampled, _ = self.buffer.sample(batch_size=len(done_indices), )
            observation_reset = batch_sampled.obs.copy()
            next_observation[done_indices] = observation_reset

        info = batch.info
        self.state = next_observation.copy()

        return next_observation, reward, done, info

    def close(self) -> None:
        pass
