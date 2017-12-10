import numpy as np
import gc


# TODO: Refactor with tf.train.slice_input_producer, tf.train.Coordinator, tf.train.QueueRunner
class Batch(object):
    """
    class for batch datasets. Collect multiple states (actions, rewards, etc.) on-policy.
    """

    def __init__(self, env, pi, adv_estimation_func): # how to name the function?
        self.env = env
        self.pi = pi
        self.adv_estimation_func = adv_estimation_func
        self.is_first_collect = True


    def collect(self, num_timesteps=0, num_episodes=0, apply_func=True): # specify how many data to collect here, or fix it in __init__()
        assert sum([num_timesteps > 0, num_episodes > 0]) == 1, "One and only one collection number specification permitted!"

        if num_timesteps > 0: # YouQiaoben: finish this implementation, the following code are just from openai/baselines
            t = 0
            ac = self.env.action_space.sample() # not used, just so we have the datatype
            new = True # marks if we're on first timestep of an episode
            if self.is_first_collect:
                ob = self.env.reset()
                self.is_first_collect = False
            else:
                ob = self.raw_data['obs'][0] # last observation!

            # Initialize history arrays
            obs = np.array([ob for _ in range(num_timesteps)])
            rews = np.zeros(num_timesteps, 'float32')
            news = np.zeros(num_timesteps, 'int32')
            acs = np.array([ac for _ in range(num_timesteps)])

            for t in range(num_timesteps):
                pass

            while True:
                prevac = ac
                ac, vpred = pi.act(stochastic, ob)
                # Slight weirdness here because we need value function at time T
                # before returning segment [0, T-1] so we get the correct
                # terminal value
                i = t % horizon
                obs[i] = ob
                vpreds[i] = vpred
                news[i] = new
                acs[i] = ac
                prevacs[i] = prevac

                ob, rew, new, _ = env.step(ac)
                rews[i] = rew

                cur_ep_ret += rew
                cur_ep_len += 1
                if new:
                    ep_rets.append(cur_ep_ret)
                    ep_lens.append(cur_ep_len)
                    cur_ep_ret = 0
                    cur_ep_len = 0
                    ob = env.reset()
                t += 1

        if num_episodes > 0: # YouQiaoben: fix memory growth, both del and gc.collect() fail
            # initialize rawdata lists
            if not self.is_first_collect:
                del self.obs
                del self.acs
                del self.rews
                del self.news

            obs = []
            acs = []
            rews = []
            news = []

            t_count = 0

            for e in range(num_episodes):
                ob = self.env.reset()
                obs.append(ob)
                news.append(True)

                while True:
                    ac = self.pi.act(ob)
                    acs.append(ac)

                    ob, rew, done, _ = self.env.step(ac)
                    rews.append(rew)

                    t_count += 1
                    if t_count >= 200: # force episode stop
                        break

                    if done: # end of episode, discard s_T
                        break
                    else:
                        obs.append(ob)
                        news.append(False)

            self.obs = np.array(obs)
            self.acs = np.array(acs)
            self.rews = np.array(rews)
            self.news = np.array(news)

            del obs
            del acs
            del rews
            del news

            self.raw_data = {'obs': self.obs, 'acs': self.acs, 'rews': self.rews, 'news': self.news}
        
            self.is_first_collect = False

        if apply_func:
            self.apply_adv_estimation_func()

        gc.collect()

    def apply_adv_estimation_func(self):
        self.data = self.adv_estimation_func(self.raw_data)

    def next_batch(self, batch_size): # YouQiaoben: referencing other iterate over batches
        rand_idx = np.random.choice(self.data['obs'].shape[0], batch_size)
        return {key: value[rand_idx] for key, value in self.data.items()}

