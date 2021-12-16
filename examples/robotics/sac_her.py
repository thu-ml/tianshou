import argparse
import os
import pprint
from functools import partial

import gym
import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import (
    Collector,
    HERCollector,
    PrioritizedReplayBuffer,
    PrioritizedVectorReplayBuffer,
    ReplayBuffer,
    VectorReplayBuffer,
)
from tianshou.env import SubprocVectorEnv
from tianshou.policy import SACHERPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic

if __name__ == '__main__':
    '''
    load param
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default='reach_fetch', help='config file name'
    )
    args = parser.parse_args()
    with open('config/' + args.config + '.yaml', "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    '''
    make env
    '''

    def make_env():
        return gym.wrappers.FlattenObservation(gym.make(config['env']))

    def make_test_env(i):
        if config['record_test']:
            return gym.wrappers.RecordVideo(
                gym.wrappers.FlattenObservation(gym.make(config['env'])),
                video_folder='log/' + config['env'] + '/video' + str(i),
                episode_trigger=lambda x: True
            )
        else:
            return gym.wrappers.FlattenObservation(gym.make(config['env']))

    env = gym.make(config['env'])
    dict_observation_space = env.observation_space
    env = gym.wrappers.FlattenObservation(env)
    obs = env.reset()
    state_shape = len(obs)
    action_shape = env.action_space.shape or env.action_space.n
    train_envs = SubprocVectorEnv(
        [make_env for _ in range(config['training_num'])], norm_obs=config['norm_obs']
    )
    if config['norm_obs']:
        print('updating env norm...')
        train_envs.reset()
        for _ in range(1000):
            _, _, done, _ = train_envs.step(
                [env.action_space.sample() for _ in range(config['training_num'])]
            )
            if np.any(done):
                env_ind = np.where(done)[0]
                train_envs.reset(env_ind)
        print('updating done!')
        train_envs.update_obs_rms = False
    test_envs = SubprocVectorEnv(
        [partial(make_test_env, i) for i in range(config['test_num'])],
        norm_obs=config['norm_obs'],
        obs_rms=train_envs.obs_rms,
        update_obs_rms=False
    )
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    train_envs.seed(config['seed'])
    test_envs.seed(config['seed'])
    '''
    build and init network
    '''
    if not (torch.cuda.is_available()):
        config['device'] = 'cpu'
    # actor
    net_a = Net(
        state_shape, hidden_sizes=config['hidden_sizes'], device=config['device']
    )
    actor = ActorProb(
        net_a,
        action_shape,
        max_action=env.action_space.high[0],
        device=config['device'],
        unbounded=True,
        conditioned_sigma=True
    ).to(config['device'])
    actor_optim = torch.optim.Adam(actor.parameters(), lr=config['actor_lr'])
    # critic
    net_c1 = Net(
        state_shape,
        action_shape,
        hidden_sizes=config['hidden_sizes'],
        concat=True,
        device=config['device']
    )
    net_c2 = Net(
        state_shape,
        action_shape,
        hidden_sizes=config['hidden_sizes'],
        concat=True,
        device=config['device']
    )
    critic1 = Critic(net_c1, device=config['device']).to(config['device'])
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=config['critic_lr'])
    critic2 = Critic(net_c2, device=config['device']).to(config['device'])
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=config['critic_lr'])
    # auto alpha
    if config['auto_alpha']:
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=config['device'])
        alpha_optim = torch.optim.Adam([log_alpha], lr=config['alpha_lr'])
        config['alpha'] = (target_entropy, log_alpha, alpha_optim)
    '''
    set up policy
    '''
    policy = SACHERPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        tau=config['tau'],
        gamma=config['gamma'],
        alpha=config['alpha'],
        estimation_step=config['estimation_step'],
        action_space=env.action_space,
        reward_normalization=False,
        dict_observation_space=dict_observation_space,
        reward_fn=env.compute_reward,
        future_k=config['replay_k'],
        strategy=config['strategy']
    )
    # load policy
    if config['resume_path']:
        policy.load_state_dict(
            torch.load(config['resume_path'], map_location=config['device'])
        )
        print("Loaded agent from: ", config['resume_path'])
    '''
    set up collector
    '''
    if config['training_num'] > 1:
        if config['use_PER']:
            buffer = PrioritizedVectorReplayBuffer(
                total_size=config['buffer_size'],
                buffer_num=len(train_envs),
                alpha=config['per_alpha'],
                beta=config['per_beta']
            )
        else:
            buffer = VectorReplayBuffer(config['buffer_size'], len(train_envs))
    else:
        if config['use_PER']:
            buffer = PrioritizedReplayBuffer(
                size=config['buffer_size'],
                alpha=config['per_alpha'],
                beta=config['per_beta']
            )
        else:
            buffer = ReplayBuffer(config['buffer_size'])
    train_collector = HERCollector(
        policy=policy,
        env=train_envs,
        buffer=buffer,
        exploration_noise=True,
        dict_observation_space=dict_observation_space,
        reward_fn=env.compute_reward,
        replay_k=config['replay_k'],
        strategy=config['strategy']
    )
    test_collector = Collector(policy, test_envs)
    # warm up
    train_collector.collect(n_step=config['start_timesteps'], random=True)
    '''
    logger
    '''
    log_file = config['info']
    log_path = os.path.join(config['logdir'], config['env'], 'sac', log_file)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(config))
    logger = TensorboardLogger(writer, update_interval=100, train_interval=100)

    # save function
    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    '''
    trainer
    '''
    # trainer
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        config['epoch'],
        config['step_per_epoch'],
        config['step_per_collect'],
        config['test_num'],
        config['batch_size'],
        save_fn=save_fn,
        logger=logger,
        update_per_step=config['update_per_step'],
        test_in_train=False,
        curriculum=config['curriculum']
    )
    pprint.pprint(result)
