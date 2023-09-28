from dataclasses import dataclass


@dataclass
class RLSamplingConfig:
    """Sampling, epochs, parallelization, buffers, collectors, and batching."""

    # TODO: What are reasonable defaults?
    num_epochs: int = 100
    step_per_epoch: int = 30000
    batch_size: int = 64
    num_train_envs: int = 64
    num_test_envs: int = 10
    buffer_size: int = 4096
    step_per_collect: int = 2048
    repeat_per_collect: int = 10
    update_per_step: int = 1
    start_timesteps: int = 0
    start_timesteps_random: bool = False
    # TODO can we set the parameters below more intelligently? Perhaps based on env. representation?
    replay_buffer_ignore_obs_next: bool = False
    replay_buffer_save_only_last_obs: bool = False
    replay_buffer_stack_num: int = 1
