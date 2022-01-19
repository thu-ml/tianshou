from typing import Any, Dict

from tianshou.data import Batch
from tianshou.data.buffer.base import ReplayBuffer
from tianshou.policy import SACPolicy


class MBPOPolicy(SACPolicy):
    """Implementation of Model-Based Policy Optimization. arXiv:1906.08253.

    MBPO builds on SAC with different training scheme.

    .. seealso::

        Please refer to :class:`~tianshou.policy.SACPolicy`
    """

    def update(
        self, env_sample_size: int, env_buffer: ReplayBuffer, model_sample_size: int,
        model_buffer: ReplayBuffer, **kwargs: Any
    ) -> Dict[str, Any]:
        """MBPO collects samples from both the environment and model rollouts."""
        env_batch, env_indice = env_buffer.sample(env_sample_size)
        model_batch, model_indice = model_buffer.sample(model_sample_size)
        self.updating = True
        env_batch = self.process_fn(env_batch, env_buffer, env_indice)
        model_batch = self.process_fn(model_batch, model_buffer, model_indice)
        batch = Batch.cat([env_batch, model_batch])
        result = self.learn(batch, **kwargs)
        env_batch = batch[:env_sample_size]
        model_batch = batch[env_sample_size:]
        self.post_process_fn(env_batch, env_buffer, env_indice)
        self.post_process_fn(model_batch, model_buffer, model_indice)
        self.updating = False
        return result
