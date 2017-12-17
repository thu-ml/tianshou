import numpy as np
import tensorflow as tf
from collections import deque
from math import fabs

from tianshou.data.replay_buffer.buffer import ReplayBuffer


class NaiveExperience(ReplayBuffer):
    def __init__(self, env, policy, qnet, target_qnet, conf):
        self.max_size = conf['size']
        self._env = env
        self._policy = policy
        self._qnet = qnet
        self._target_qnet = target_qnet
        self._begin_act()
        self.n_entries = 0
        self.memory = deque(maxlen=self.max_size)

    def add(self, data, priority=0):
        self.memory.append(data)
        if self.n_entries < self.max_size:
            self.n_entries += 1

    def _begin_act(self):
        """
        if the previous interaction is ended or the interaction hasn't started
        then begin act from the state of env.reset()
        """
        self.observation = self._env.reset()
        self.action = self._env.action_space.sample()
        done = False
        while not done:
            if done:
                self.observation = self._env.reset()
                self.action = self._env.action_space.sample()
            self.observation, _, done, _ = self._env.step(self.action)

    def collect(self):
        """
        collect data for replay memory and update the priority according to the given data.
        store the previous action, previous observation, reward, action, observation in the replay memory.
        """
        sess = tf.get_default_session()
        current_data = dict()
        current_data['previous_action'] = self.action
        current_data['previous_observation'] = self.observation
        self.action = np.argmax(sess.run(self._policy, feed_dict={"dqn_observation:0": self.observation.reshape((1,) + self.observation.shape)}))
        self.observation, reward, done, _ = self._env.step(self.action)
        current_data['action'] = self.action
        current_data['observation'] = self.observation
        current_data['reward'] = reward
        self.add(current_data)
        if done:
            self._begin_act()

    def update_priority(self, indices, priorities=0):
        pass

    def reset_alpha(self, alpha):
        pass

    def sample(self, conf):
        batch_size = conf['batch_size']
        batch_size = min(len(self.memory), batch_size)
        idxs = np.random.choice(len(self.memory), batch_size)
        return [self.memory[idx] for idx in idxs], [1] * len(idxs), idxs

    def next_batch(self, batch_size):
        """
        collect a batch of data from replay buffer, update the priority and calculate the necessary statistics for
        updating q value network.
        :param batch_size: int batch size.
        :return: a batch of data, with target storing the target q value and wi, rewards storing the coefficient
        for gradient of q value network.
        """
        data = dict()
        observations = list()
        actions = list()
        rewards = list()
        wi = list()
        target = list()

        for i in range(0, batch_size):
            current_datas, current_wis, current_indexs = self.sample({'batch_size': 1})
            current_data = current_datas[0]
            current_wi = current_wis[0]
            current_index = current_indexs[0]
            observations.append(current_data['observation'])
            actions.append(current_data['action'])
            next_max_qvalue = np.max(self._target_qnet.values(current_data['observation']))
            current_qvalue = self._qnet.values(current_data['previous_observation'])[0, current_data['previous_action']]
            reward = current_data['reward'] + next_max_qvalue - current_qvalue
            rewards.append(reward)
            target.append(current_data['reward'] + next_max_qvalue)
            self.update_priority(current_index, [fabs(reward)])
            wi.append(current_wi)

        data['observations'] = np.array(observations)
        data['actions'] = np.array(actions)
        data['rewards'] = np.array(rewards)
        data['wi'] = np.array(wi)
        data['target'] = np.array(target)

        return data

    def rebalance(self):
        pass
