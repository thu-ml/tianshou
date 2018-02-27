import numpy as np


def full_return(raw_data):
    """
    naively compute full return
    :param raw_data: dict of specified keys and values.
    """
    observations = raw_data['observation']
    actions = raw_data['action']
    rewards = raw_data['reward']
    episode_start_flags = raw_data['end_flag']
    num_timesteps = rewards.shape[0]

    data = {}

    returns = rewards.copy()
    episode_start_idx = 0
    for i in range(1, num_timesteps):
        if episode_start_flags[i] or (
                i == num_timesteps - 1):  # found the start of next episode or the end of all episodes
            if i < rewards.shape[0] - 1:
                t = i - 1
            else:
                t = i
            Gt = 0
            while t >= episode_start_idx:
                Gt += rewards[t]
                returns[t] = Gt
                t -= 1

            episode_start_idx = i

    data['return'] = returns

    return data


class gae_lambda:
    """
    Generalized Advantage Estimation (Schulman, 15) to compute advantage
    """
    def __init__(self, T, value_function):
        self.T = T
        self.value_function = value_function

    def __call__(self, raw_data):
        reward = raw_data['reward']
        observation = raw_data['observation']

        state_value = self.value_function.eval_value(observation)

        # wrong version of advantage just to run
        advantage = reward + state_value

        return {'advantage': advantage}


class nstep_return:
    """
    compute the n-step return from n-step rewards and bootstrapped value function
    """
    def __init__(self, n, value_function):
        self.n = n
        self.value_function = value_function

    def __call__(self, raw_data):
        reward = raw_data['reward']
        observation = raw_data['observation']

        state_value = self.value_function.eval_value(observation)

        # wrong version of return just to run
        return_ = reward + state_value

        return {'return': return_}


class ddpg_return:
    """
    compute the return as in DDPG. this seems to have to be special
    """
    def __init__(self, actor, critic, use_target_network=True):
        self.actor = actor
        self.critic = critic
        self.use_target_network = use_target_network

    def __call__(self, raw_data):
        observation = raw_data['observation']
        reward = raw_data['reward']

        if self.use_target_network:
            action_target = self.actor.eval_action_old(observation)
            value_target = self.critic.eval_value_old(observation, action_target)
        else:
            action_target = self.actor.eval_action(observation)
            value_target = self.critic.eval_value(observation, action_target)

        return_ = reward + value_target

        return {'return': return_}


class nstep_q_return:
    """
    compute the n-step return for Q-learning targets
    """
    def __init__(self, n, action_value, use_target_network=True):
        self.n = n
        self.action_value = action_value
        self.use_target_network = use_target_network

    def __call__(self, raw_data):
        # raw_data should contain 'next_observation' from replay memory...?
        # maybe the main difference between Batch and Replay is the stored data format?
        reward = raw_data['reward']
        observation = raw_data['observation']

        if self.use_target_network:
            action_value_all_actions = self.action_value.eval_value_all_actions_old(observation)
        else:
            action_value_all_actions = self.action_value.eval_value_all_actions(observation)

        action_value_max = np.max(action_value_all_actions, axis=1)

        return_ = reward + action_value_max

        return {'return': return_}


class QLearningTarget:
    def __init__(self, policy, gamma):
        self._policy = policy
        self._gamma = gamma

    def __call__(self, raw_data):
        data = dict()
        observations = list()
        actions = list()
        rewards = list()
        wi = list()
        all_data, data_wi, data_index = raw_data

        for i in range(0, all_data.shape[0]):
            current_data = all_data[i]
            current_wi = data_wi[i]
            current_index = data_index[i]
            observations.append(current_data['observation'])
            actions.append(current_data['action'])
            next_max_qvalue = np.max(self._policy.values(current_data['observation']))
            current_qvalue = self._policy.values(current_data['previous_observation'])[current_data['previous_action']]
            reward = current_data['reward'] + next_max_qvalue - current_qvalue
            rewards.append(reward)
            wi.append(current_wi)

        data['observations'] = np.array(observations)
        data['actions'] = np.array(actions)
        data['rewards'] = np.array(rewards)

        return data


class ReplayMemoryQReturn:
    """
    compute the n-step return for Q-learning targets
    """
    def __init__(self, n, action_value, use_target_network=True):
        self.n = n
        self._action_value = action_value
        self._use_target_network = use_target_network

    def __call__(self, raw_data):
        reward = raw_data['reward']
        observation = raw_data['observation']

        if self._use_target_network:
            # print(observation.shape)
            # print((observation.reshape((1,) + observation.shape)))
            action_value_all_actions = self._action_value.eval_value_all_actions_old(observation.reshape((1,) + observation.shape))
        else:
            # print(observation.shape)
            # print((observation.reshape((1,) + observation.shape)))
            action_value_all_actions = self._action_value.eval_value_all_actions(observation.reshape((1,) + observation.shape))

        action_value_max = np.max(action_value_all_actions, axis=1)

        return_ = reward + action_value_max

        return {'return': return_}
