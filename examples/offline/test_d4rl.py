import d4rl
import gym

env = gym.make("halfcheetah-medium-v1")
dataset = d4rl.qlearning_dataset(env)