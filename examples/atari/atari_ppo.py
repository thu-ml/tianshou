#!/usr/bin/env python3

import datetime
import os
import pprint
import sys
from collections.abc import Sequence
from typing import cast

import numpy as np
import torch
from sensai.util import logging

from tianshou.algorithm import PPO
from tianshou.algorithm.algorithm_base import Algorithm
from tianshou.algorithm.modelbased.icm import ICMOnPolicyWrapper
from tianshou.algorithm.modelfree.reinforce import DiscreteActorPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory, LRSchedulerFactoryLinear
from tianshou.data import Collector, CollectStats, VectorReplayBuffer
from tianshou.env.atari.atari_network import (
    DQNet,
    ScaledObsInputActionReprNet,
    layer_init,
)
from tianshou.env.atari.atari_wrapper import make_atari_env
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.trainer import OnPolicyTrainerParams
from tianshou.utils.net.discrete import (
    DiscreteActor,
    DiscreteCritic,
    IntrinsicCuriosityModule,
)

log = logging.getLogger(__name__)


def main(
    task: str = "PongNoFrameskip-v4",
    seed: int = 4213,
    scale_obs: int = 1,
    buffer_size: int = 100000,
    lr: float = 2.5e-4,
    gamma: float = 0.99,
    epoch: int = 100,
    epoch_num_steps: int = 100000,
    collection_step_num_env_steps: int = 1000,
    update_step_num_repetitions: int = 4,
    batch_size: int = 256,
    hidden_size: int = 512,
    num_training_envs: int = 10,
    num_test_envs: int = 10,
    return_scaling: bool = False,
    vf_coef: float = 0.25,
    ent_coef: float = 0.01,
    gae_lambda: float = 0.95,
    lr_decay: int = True,
    max_grad_norm: float = 0.5,
    eps_clip: float = 0.1,
    dual_clip: float | None = None,
    value_clip: bool = True,
    advantage_normalization: bool = True,
    recompute_adv: bool = False,
    persistence_base_dir: str = "log",
    render: float = 0.0,
    device: str | None = None,
    frames_stack: int = 4,
    resume_path: str | None = None,
    resume_id: str | None = None,
    logger_type: str = "tensorboard",
    wandb_project: str = "atari.benchmark",
    watch: bool = False,
    save_buffer_name: str | None = None,
    icm_lr_scale: float = 0.0,
    icm_reward_scale: float = 0.01,
    icm_forward_loss_weight: float = 0.2,
) -> None:
    # Set defaults for mutable arguments
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get all local variables as config (excluding internal/temporary ones)
    params_log_info = locals()
    log.info(f"Starting training with config:\n{params_log_info}")

    state_shape: tuple[int, ...] | int
    action_shape: Sequence[int] | int

    env, training_envs, test_envs = make_atari_env(
        task,
        seed,
        num_training_envs,
        num_test_envs,
        scale=scale_obs,
        frame_stack=frames_stack,
    )
    state_shape = cast(tuple[int, ...], env.observation_space.shape)
    action_shape = cast(Sequence[int] | int, env.action_space.shape or env.action_space.n)  # type: ignore
    # should be N_FRAMES x H x W
    log.info(f"Observations shape: {state_shape}")
    log.info(f"Actions shape: {action_shape}")
    # seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    # define model
    c, h, w = state_shape
    net: ScaledObsInputActionReprNet | DQNet
    net = DQNet(
        c=c,
        h=h,
        w=w,
        action_shape=action_shape,
        features_only=True,
        output_dim_added_layer=hidden_size,
        layer_init=layer_init,
    )
    if scale_obs:
        net = ScaledObsInputActionReprNet(net)
    actor = DiscreteActor(preprocess_net=net, action_shape=action_shape, softmax_output=False)
    critic = DiscreteCritic(preprocess_net=net)
    optim = AdamOptimizerFactory(lr=lr, eps=1e-5)

    if lr_decay:
        optim.with_lr_scheduler_factory(
            LRSchedulerFactoryLinear(
                max_epochs=epoch,
                epoch_num_steps=epoch_num_steps,
                collection_step_num_env_steps=collection_step_num_env_steps,
            )
        )

    # define algorithm
    policy = DiscreteActorPolicy(
        actor=actor,
        action_space=env.action_space,
    )
    algorithm: PPO = PPO(
        policy=policy,
        critic=critic,
        optim=optim,
        gamma=gamma,
        gae_lambda=gae_lambda,
        max_grad_norm=max_grad_norm,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        return_scaling=return_scaling,
        eps_clip=eps_clip,
        value_clip=value_clip,
        dual_clip=dual_clip,
        advantage_normalization=advantage_normalization,
        recompute_advantage=recompute_adv,
    ).to(device)
    if icm_lr_scale > 0:
        c, h, w = state_shape
        feature_net = DQNet(c=c, h=h, w=w, action_shape=action_shape, features_only=True)
        action_dim = int(np.prod(action_shape))
        feature_dim = feature_net.output_dim
        icm_net = IntrinsicCuriosityModule(
            feature_net=feature_net.net,
            feature_dim=feature_dim,
            action_dim=action_dim,
            hidden_sizes=[hidden_size],
        )
        icm_optim = AdamOptimizerFactory(lr=lr)
        algorithm = ICMOnPolicyWrapper(  # type: ignore[assignment]
            wrapped_algorithm=algorithm,
            model=icm_net,
            optim=icm_optim,
            lr_scale=icm_lr_scale,
            reward_scale=icm_reward_scale,
            forward_loss_weight=icm_forward_loss_weight,
        ).to(device)
    # load a previous policy
    if resume_path:
        algorithm.load_state_dict(torch.load(resume_path, map_location=device))
        log.info(f"Loaded agent from: {resume_path}")
    # replay buffer: `save_last_obs` and `stack_num` can be removed together
    # when you have enough RAM
    buffer = VectorReplayBuffer(
        buffer_size,
        buffer_num=len(training_envs),
        ignore_obs_next=True,
        save_only_last_obs=True,
        stack_num=frames_stack,
    )
    # collector
    training_collector = Collector[CollectStats](
        algorithm, training_envs, buffer, exploration_noise=True
    )
    test_collector = Collector[CollectStats](algorithm, test_envs, exploration_noise=True)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    algo_name = "ppo_icm" if icm_lr_scale > 0 else "ppo"
    log_name = os.path.join(task, algo_name, str(seed), now)
    log_path = os.path.join(persistence_base_dir, log_name)

    # logger
    logger_factory = LoggerFactoryDefault()
    if logger_type == "wandb":
        logger_factory.logger_type = "wandb"
        logger_factory.wandb_project = wandb_project
    else:
        logger_factory.logger_type = "tensorboard"

    logger = logger_factory.create_logger(
        log_dir=log_path,
        experiment_name=log_name,
        run_id=resume_id,
        config_dict=params_log_info,
    )

    def save_best_fn(policy: Algorithm) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards: float) -> bool:
        if env.spec.reward_threshold:  # type: ignore
            return mean_rewards >= env.spec.reward_threshold  # type: ignore
        if "Pong" in task:
            return mean_rewards >= 20
        return False

    def save_checkpoint_fn(epoch: int, env_step: int, gradient_step: int) -> str:
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
        torch.save({"model": algorithm.state_dict()}, ckpt_path)
        return ckpt_path

    def watch_fn() -> None:
        log.info("Setup test envs ...")
        test_envs.seed(seed)
        if save_buffer_name:
            log.info(f"Generate buffer with size {buffer_size}")
            buffer = VectorReplayBuffer(
                buffer_size,
                buffer_num=len(test_envs),
                ignore_obs_next=True,
                save_only_last_obs=True,
                stack_num=frames_stack,
            )
            collector = Collector[CollectStats](
                algorithm, test_envs, buffer, exploration_noise=True
            )
            result = collector.collect(n_step=buffer_size, reset_before_collect=True)
            log.info(f"Save buffer into {save_buffer_name}")
            # Unfortunately, pickle will cause oom with 1M buffer size
            buffer.save_hdf5(save_buffer_name)
        else:
            log.info("Testing agent ...")
            test_collector.reset()
            result = test_collector.collect(n_episode=num_test_envs, render=render)
        result.pprint_asdict()

    if watch:
        watch_fn()
        sys.exit(0)

    # test training_collector and start filling replay buffer
    training_collector.reset()
    training_collector.collect(n_step=batch_size * num_training_envs)

    # train
    result = algorithm.run_training(
        OnPolicyTrainerParams(
            training_collector=training_collector,
            test_collector=test_collector,
            max_epochs=epoch,
            epoch_num_steps=epoch_num_steps,
            update_step_num_repetitions=update_step_num_repetitions,
            test_step_num_episodes=num_test_envs,
            batch_size=batch_size,
            collection_step_num_env_steps=collection_step_num_env_steps,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            test_in_training=False,
            resume_from_log=resume_id is not None,
            save_checkpoint_fn=save_checkpoint_fn,
        )
    )

    pprint.pprint(result)
    watch_fn()


if __name__ == "__main__":
    result = logging.run_cli(main, level=logging.INFO)
