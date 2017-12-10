from buffer import ReplayBuffer
import numpy as np
from collections import deque

class NaiveExperience(ReplayBuffer):
	def __init__(self, conf):
		self.max_size = conf['size']
		self.n_entries = 0
		self.memory = deque(maxlen = self.max_size)

	def add(self, data, priority = 0):
		self.memory.append(data)
		if self.n_entries < self.max_size:
			self.n_entries += 1

	def update_priority(self, indices, priorities = 0):
		pass

	def reset_alpha(self, alpha):
		pass

	def sample(self, conf):
		batch_size = conf['batch_size']
		batch_size = min(len(self.memory), batch_size)
		idxs = np.random.choice(len(self.memory), batch_size)
		return [self.memory[idx] for idx in idxs], [1] * len(idxs), idxs

	def rebalance(self):
		pass
