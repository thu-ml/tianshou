Train a model-free RL agent within 30s
======================================

This page summarizes some hyper-parameter tuning experience and code-level trick when training a model-free DRL agent.

You can also contribute to this page with your own tricks :)


Avoid batch-size = 1
--------------------

In the traditional RL training loop, we always use the policy to interact with only one environment for collecting data. That means most of the time the network use batch-size = 1. Quite inefficient!
Here is an example of showing how inefficient it is:
::

    import torch, time
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(3, 128), nn.ReLU(inplace=True),
                nn.Linear(128, 128), nn.ReLU(inplace=True),
                nn.Linear(128, 1))
        def forward(self, s):
            return self.model(s)

    net = Net()
    cnt = 1000
    div = 128
    a = torch.randn([128, 3])

    t = time.time()
    for i in range(cnt):
        b = net(a)
    t1 = (time.time() - t) / cnt
    print(t1)
    t = time.time()
    for i in range(cnt):
        for a_ in a.split(a.shape[0] // div):
            b = net(a_)
    t2 = (time.time() - t) / cnt
    print(t2)
    print(t2 / t1)

The first test uses batch-size 128, and the second test uses batch-size = 1 for 128 times. In our test, the first is 70-80 times faster than the second.

So how could we avoid the case of batch-size = 1? The answer is synchronize sampling: we create multiple independent environments and sample simultaneously. It is similar to A2C, but other algorithms can also use this method. In our experiments, sampling from more environments benefits not only the sample speed but also the converge speed of neural network (we guess it lowers the sample bias).

By the way, A2C is better than A3C in some cases: A3C needs to act independently and sync the gradient to master, but, in a single node, using A3C to act with batch-size = 1 is quite resource-consuming.


Algorithm specific tricks
-------------------------

Here is about the experience of hyper-parameter tuning on CartPole and Pendulum:

* :class:`~tianshou.policy.DQNPolicy`: use estimation_step greater than 1 and target network, also with a suitable size of replay buffer;
* :class:`~tianshou.policy.PGPolicy`: TBD
* :class:`~tianshou.policy.A2CPolicy`: TBD
* :class:`~tianshou.policy.PPOPolicy`: TBD
* :class:`~tianshou.policy.DDPGPolicy`, :class:`~tianshou.policy.TD3Policy`, and :class:`~tianshou.policy.SACPolicy`: We found two tricks. The first is to ignore the done flag. The second is to normalize reward to a standard normal distribution (it is against the theoretical analysis, but indeed works very well). The two tricks work amazingly on Mujoco tasks, typically with a faster converge speed (1M -> 200K).

* On-policy algorithms: increase the repeat-time (to 2 or 4 for trivial benchmark, 10 for mujoco) of the given batch in each training update will make the algorithm more stable. 


Code-level optimization
-----------------------

Tianshou has many short-but-efficient lines of code. For example, when we want to compute :math:`V(s)` and :math:`V(s')` by the same network, the best way is to concatenate :math:`s` and :math:`s'` together instead of computing the value function using twice of network forward.

.. Jiayi: I write each line of code after quite a lot of time of consideration. Details make a difference.


Finally
-------

With fast-speed sampling, we could use large batch-size and large learning rate for faster convergence.

RL algorithms are seed-sensitive. Try more seeds and pick the best. But for our demo, we just used seed = 0 and found it work surprisingly well on policy gradient, so we did not try other seed.

.. image:: /_static/images/testpg.gif
    :align: center
