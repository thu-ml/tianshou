from .replay_buffer.base import ReplayBufferBase

class DataCollector(object):
    """
    a utility class to manage the interaction between buffer and advantage_estimation
    """
    def __init__(self, env, policy, data_buffer, process_functions, managed_networks):
        self.env = env
        self.policy = policy
        self.data_buffer = data_buffer
        self.process_functions = process_functions
        self.managed_networks = managed_networks

        self.required_placeholders = {}
        for net in self.managed_networks:
            self.required_placeholders.update(net.managed_placeholders)
        self.require_advantage = 'advantage' in self.required_placeholders.keys()

        if isinstance(self.data_buffer, ReplayBufferBase):  # process when sampling minibatch
            self.process_mode = 'minibatch'
        else:
            self.process_mode = 'batch'

        self.current_observation = self.env.reset()

    def collect(self, num_timesteps=1, num_episodes=0, exploration=None, my_feed_dict={}):
        assert sum([num_timesteps > 0, num_episodes > 0]) == 1,\
            "One and only one collection number specification permitted!"

        if num_timesteps > 0:
            for _ in range(num_timesteps):
                action_vanilla = self.policy.act(self.current_observation, my_feed_dict=my_feed_dict)
                if exploration:
                    action = exploration(action_vanilla)
                else:
                    action = action_vanilla

                next_observation, reward, done, _ = self.env.step(action)
                self.data_buffer.add((self.current_observation, action, reward, done))
                self.current_observation = next_observation

        if num_episodes > 0:
            for _ in range(num_episodes):
                observation = self.env.reset()
                done = False
                while not done:
                    action_vanilla = self.policy.act(observation, my_feed_dict=my_feed_dict)
                    if exploration:
                        action = exploration(action_vanilla)
                    else:
                        action = action_vanilla

                    next_observation, reward, done, _ = self.env.step(action)
                    self.data_buffer.add((observation, action, reward, done))
                    observation = next_observation

    def next_batch(self, batch_size):
        sampled_index = self.data_buffer.sample(batch_size)
        if self.process_mode == 'minibatch':
            pass

        # flatten rank-2 list to numpy array

        return

    def statistics(self):
        pass