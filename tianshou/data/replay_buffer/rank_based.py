#!/usr/bin/python
# -*- encoding=utf-8 -*-
# author: Ian
# e-mail: stmayue@gmail.com
# description: 

import sys
import math
import random
import numpy as np
import tensorflow as tf

from tianshou.data.replay_buffer.binary_heap import BinaryHeap
from tianshou.data.replay_buffer.buffer import ReplayBuffer


class RankBasedExperience(ReplayBuffer):

    def __init__(self, env, policy, qnet, target_qnet, conf):
        self.size = conf['size']
        self.replace_flag = conf['replace_old'] if 'replace_old' in conf else True
        self.priority_size = conf['priority_size'] if 'priority_size' in conf else self.size

        self.alpha = conf['alpha'] if 'alpha' in conf else 0.7
        self.beta_zero = conf['beta_zero'] if 'beta_zero' in conf else 0.5
        self.batch_size = conf['batch_size'] if 'batch_size' in conf else 32
        self.learn_start = conf['learn_start'] if 'learn_start' in conf else 1000
        self.total_steps = conf['steps'] if 'steps' in conf else 100000
        # partition number N, split total size to N part
        self.partition_num = conf['partition_num'] if 'partition_num' in conf else 10

        self.index = 0
        self.record_size = 0
        self.isFull = False

        self._env = env
        self._policy = policy
        self._qnet = qnet
        self._target_qnet = target_qnet
        self._begin_act()

        self._experience = {}
        self.priority_queue = BinaryHeap(self.priority_size)
        self.distributions = self.build_distributions()

        self.beta_grad = (1 - self.beta_zero) / (self.total_steps - self.learn_start)

    def build_distributions(self):
        """
        preprocess pow of rank
        (rank i) ^ (-alpha) / sum ((rank i) ^ (-alpha))
        :return: distributions, dict
        """
        res = {}
        n_partitions = self.partition_num
        partition_num = 1
        # each part size
        partition_size = int(math.floor(self.size / n_partitions))

        for n in range(partition_size, self.size + 1, partition_size):
            if self.learn_start <= n <= self.priority_size:
                distribution = {}
                # P(i) = (rank i) ^ (-alpha) / sum ((rank i) ^ (-alpha))
                pdf = list(
                    map(lambda x: math.pow(x, -self.alpha), range(1, n + 1))
                )
                pdf_sum = math.fsum(pdf)
                distribution['pdf'] = list(map(lambda x: x / pdf_sum, pdf))
                # split to k segment, and than uniform sample in each k
                # set k = batch_size, each segment has total probability is 1 / batch_size
                # strata_ends keep each segment start pos and end pos
                cdf = np.cumsum(distribution['pdf'])
                strata_ends = {1: 0, self.batch_size + 1: n}
                step = 1. / self.batch_size
                index = 1
                for s in range(2, self.batch_size + 1):
                    while cdf[index] < step:
                        index += 1
                    strata_ends[s] = index
                    step += 1. / self.batch_size

                distribution['strata_ends'] = strata_ends

                res[partition_num] = distribution

            partition_num += 1

        return res

    def fix_index(self):
        """
        get next insert index
        :return: index, int
        """
        if self.record_size <= self.size:
            self.record_size += 1
        if self.index % self.size == 0:
            self.isFull = True if len(self._experience) == self.size else False
            if self.replace_flag:
                self.index = 1
                return self.index
            else:
                sys.stderr.write('Experience replay buff is full and replace is set to FALSE!\n')
                return -1
        else:
            self.index += 1
            return self.index

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

        sess = tf.get_default_session()
        # TODO: pre-build the thing in sess.run
        current_datas, current_wis, current_indexs = self.sample({'global_step': sess.run(tf.train.get_global_step())})

        for i in range(0, batch_size):
            current_data = current_datas[i]
            current_wi = current_wis[i]
            current_index = current_indexs[i]
            observations.append(current_data['observation'])
            actions.append(current_data['action'])
            next_max_qvalue = np.max(self._target_qnet.values(current_data['observation']))
            current_qvalue = self._qnet.values(current_data['previous_observation'])[0, current_data['previous_action']]
            reward = current_data['reward'] + next_max_qvalue - current_qvalue
            rewards.append(reward)
            target.append(current_data['reward'] + next_max_qvalue)
            self.update_priority([current_index], [math.fabs(reward)])
            wi.append(current_wi)

        data['observations'] = np.array(observations)
        data['actions'] = np.array(actions)
        data['rewards'] = np.array(rewards)
        data['wi'] = np.array(wi)
        data['target'] = np.array(target)

        return data

    def add(self, data, priority = 1):
        """
        store experience, suggest that experience is a tuple of (s1, a, r, s2, t)
        so each experience is valid
        :param experience: maybe a tuple, or list
        :return: bool, indicate insert status
        """
        insert_index = self.fix_index()
        if insert_index > 0:
            if insert_index in self._experience:
                del self._experience[insert_index]
            self._experience[insert_index] = data
            # add to priority queue
            priority = self.priority_queue.get_max_priority()
            self.priority_queue.update(priority, insert_index)
            return True
        else:
            sys.stderr.write('Insert failed\n')
            return False

    def retrieve(self, indices):
        """
        get experience from indices
        :param indices: list of experience id
        :return: experience replay sample
        """
        return [self._experience[v] for v in indices]

    def rebalance(self):
        """
        rebalance priority queue
        :return: None
        """
        self.priority_queue.balance_tree()

    def update_priority(self, indices, delta):
        """
        update priority according indices and deltas
        :param indices: list of experience id
        :param delta: list of delta, order correspond to indices
        :return: None
        """
        for i in range(0, len(indices)):
            self.priority_queue.update(math.fabs(delta[i]), indices[i])

    def sample(self, conf):
        """
        sample a mini batch from experience replay
        :param global_step: now training step
        :return: experience, list, samples
        :return: w, list, weights
        :return: rank_e_id, list, samples id, used for update priority
        """
        global_step = conf['global_step']
        if self.record_size < self.learn_start:
            sys.stderr.write('Record size less than learn start! Sample failed\n')
            return False, False, False

        dist_index = math.floor(self.record_size * 1. / self.size * self.partition_num)
        # issue 1 by @camigord
        partition_size = math.floor(self.size * 1. / self.partition_num)
        partition_max = dist_index * partition_size
        distribution = self.distributions[dist_index]
        rank_list = []
        # sample from k segments
        for n in range(1, self.batch_size + 1):
            index = random.randint(distribution['strata_ends'][n],
                                       distribution['strata_ends'][n + 1])
            rank_list.append(index)

        # beta, increase by global_step, max 1
        beta = min(self.beta_zero + (global_step - self.learn_start - 1) * self.beta_grad, 1)
        # find all alpha pow, notice that pdf is a list, start from 0
        alpha_pow = [distribution['pdf'][v - 1] for v in rank_list]
        # w = (N * P(i)) ^ (-beta) / max w
        w = np.power(np.array(alpha_pow) * partition_max, -beta)
        w_max = max(w)
        w = np.divide(w, w_max)
        # rank list is priority id
        # convert to experience id
        rank_e_id = self.priority_queue.priority_to_experience(rank_list)
        # get experience id according rank_e_id
        experience = self.retrieve(rank_e_id)
        return experience, w, rank_e_id
