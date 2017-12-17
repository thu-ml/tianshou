class ReplayBuffer(object):
    def __init__(self, env, policy, qnet, target_qnet, conf):
        """
		Initialize a replay buffer with parameters in conf.
		"""
        pass

    def add(self, data, priority):
        """
		Add a data with priority = priority to replay buffer.
		"""
        pass

    def collect(self):
        """
		Collect data from current environment and policy.
		"""
        pass

    def next_batch(self, batch_size):
        """
		get batch of data from the replay buffer.
		"""
        pass

    def update_priority(self, indices, priorities):
        """
		Update the data's priority whose indices = indices.
		For proportional replay buffer, the priority is the priority.
		For rank based replay buffer, the priorities parameter will be the delta used to update the priority.
		"""
        pass

    def reset_alpha(self, alpha):
        """
		This function only works for proportional replay buffer.
		This function resets alpha.
		"""
        pass

    def sample(self, conf):
        """
		Sample from replay buffer with parameters in conf.
		"""
        pass

    def rebalance(self):
        """
		This is for rank based priority replay buffer, which is used to rebalance the sum tree of the priority queue.
		"""
        pass
