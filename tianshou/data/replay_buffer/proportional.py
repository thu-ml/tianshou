import numpy as np
import random
import tensorflow as tf
import math

from tianshou.data.replay_buffer import sum_tree
from tianshou.data.replay_buffer.buffer import ReplayBuffer


class PropotionalExperience(ReplayBuffer):
    """ The class represents prioritized experience replay buffer.

    The class has functions: store samples, pick samples with 
    probability in proportion to sample's priority, update 
    each sample's priority, reset alpha.

    see https://arxiv.org/pdf/1511.05952.pdf .

    """
    
    def __init__(self, env, policy, qnet, target_qnet, conf):
        """ Prioritized experience replay buffer initialization.
        
        Parameters
        ----------
        memory_size : int
            sample size to be stored
        batch_size : int
            batch size to be selected by `select` method
        alpha: float
            exponent determine how much prioritization.
            Prob_i \sim priority_i**alpha/sum(priority**alpha)
        """
        memory_size = conf['size']
        batch_size = conf['batch_size']
        alpha = conf['alpha'] if 'alpha' in conf else 0.6
        self.tree = sum_tree.SumTree(memory_size)
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.alpha = alpha
        self._env = env
        self._policy = policy
        self._qnet = qnet
        self._target_qnet = target_qnet
        self._begin_act()

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

    def add(self, data, priority):
        """ Add new sample.
        
        Parameters
        ----------
        data : object
            new sample
        priority : float
            sample's priority
        """
        self.tree.add(data, priority**self.alpha)

    def sample(self, conf):
        """ The method return samples randomly.
        
        Parameters
        ----------
        beta : float
        
        Returns
        -------
        out : 
            list of samples
        weights: 
            list of weight
        indices:
            list of sample indices
            The indices indicate sample positions in a sum tree.
            :param conf: giving beta
        """
        beta = conf['beta'] if 'beta' in conf else 0.4
        if self.tree.filled_size() < self.batch_size:
            return None, None, None

        out = []
        indices = []
        weights = []
        priorities = []
        for _ in range(self.batch_size):
            r = random.random()
            data, priority, index = self.tree.find(r)
            priorities.append(priority)
            weights.append((1./self.memory_size/priority)**beta if priority > 1e-16 else 0)
            indices.append(index)
            out.append(data)
            self.update_priority([index], [0]) # To avoid duplicating
            
        
        self.update_priority(indices, priorities) # Revert priorities

        max_weights = max(weights)

        weights[:] = [x / max_weights for x in weights] # Normalize for stability
        
        return out, weights, indices

    def collect(self):
        """
        collect data for replay memory and update the priority according to the given data.
        store the previous action, previous observation, reward, action, observation in the replay memory.
        """
        sess = tf.get_default_session()
        current_data = dict()
        current_data['previous_action'] = self.action
        current_data['previous_observation'] = self.observation
        # TODO: change the name of the feed_dict
        self.action = np.argmax(sess.run(self._policy, feed_dict={"dqn_observation:0": self.observation.reshape((1,) + self.observation.shape)}))
        self.observation, reward, done, _ = self._env.step(self.action)
        current_data['action'] = self.action
        current_data['observation'] = self.observation
        current_data['reward'] = reward
        priorities = np.array([self.tree.get_val(i) ** -self.alpha for i in range(self.tree.filled_size())])
        priority = np.max(priorities) if len(priorities) > 0 else 1
        self.add(current_data, priority)
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
            self.update_priority([current_index], [math.fabs(reward)])
            wi.append(current_wi)

        data['observations'] = np.array(observations)
        data['actions'] = np.array(actions)
        data['rewards'] = np.array(rewards)
        data['wi'] = np.array(wi)
        data['target'] = np.array(target)

        return data

    def update_priority(self, indices, priorities):
        """ The methods update samples's priority.
        
        Parameters
        ----------
        indices : 
            list of sample indices
        """
        for i, p in zip(indices, priorities):
            self.tree.val_update(i, p**self.alpha)
    
    def reset_alpha(self, alpha):
        """ Reset a exponent alpha.

        Parameters
        ----------
        alpha : float
        """
        self.alpha, old_alpha = alpha, self.alpha
        priorities = [self.tree.get_val(i)**-old_alpha for i in range(self.tree.filled_size())]
        self.update_priority(range(self.tree.filled_size()), priorities)

        
            
        
        
