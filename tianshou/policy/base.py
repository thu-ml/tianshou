import numpy as np
from torch import nn
from abc import ABC, abstractmethod


class BasePolicy(ABC, nn.Module):
    """Tianshou aims to modularizing RL algorithms. It comes into several
    classes of policies in Tianshou. All of the policy classes must inherit
    :class:`~tianshou.policy.BasePolicy`.

    A policy class typically has four parts:

    * :meth:`~tianshou.policy.BasePolicy.__init__`: initialize the policy, \
        including coping the target network and so on;
    * :meth:`~tianshou.policy.BasePolicy.forward`: compute action with given \
        observation;
    * :meth:`~tianshou.policy.BasePolicy.process_fn`: pre-process data from \
        the replay buffer (this function can interact with replay buffer);
    * :meth:`~tianshou.policy.BasePolicy.learn`: update policy with a given \
        batch of data.

    Most of the policy needs a neural network to predict the action and an
    optimizer to optimize the policy. The rules of self-defined networks are:

    1. Input: observation ``obs`` (may be a ``numpy.ndarray`` or \
        ``torch.Tensor``), hidden state ``state`` (for RNN usage), and other \
        information ``info`` provided by the environment.
    2. Output: some ``logits`` and the next hidden state ``state``. The logits\
        could be a tuple instead of a ``torch.Tensor``. It depends on how the \
        policy process the network output. For example, in PPO, the return of \
        the network might be ``(mu, sigma), state`` for Gaussian policy.

    Since :class:`~tianshou.policy.BasePolicy` inherits ``torch.nn.Module``,
    you can use :class:`~tianshou.policy.BasePolicy` almost the same as
    ``torch.nn.Module``, for instance, loading and saving the model:
    ::

        torch.save(policy.state_dict(), 'policy.pth')
        policy.load_state_dict(torch.load('policy.pth'))
    """

    def __init__(self, **kwargs):
        super().__init__()

    def process_fn(self, batch, buffer, indice):
        """Pre-process the data from the provided replay buffer. Check out
        :ref:`policy_concept` for more information.
        """
        return batch

    @abstractmethod
    def forward(self, batch, state=None, **kwargs):
        """Compute action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` which MUST have the following\
        keys:

            * ``act`` an numpy.ndarray or a torch.Tensor, the action over \
                given batch data.
            * ``state`` a dict, an numpy.ndarray or a torch.Tensor, the \
                internal state of the policy, ``None`` as default.

        Other keys are user-defined. It depends on the algorithm. For example,
        ::

            # some code
            return Batch(logits=..., act=..., state=None, dist=...)
        """
        pass

    @abstractmethod
    def learn(self, batch, **kwargs):
        """Update policy with a given batch of data.

        :return: A dict which includes loss and its corresponding label.
        """
        pass

    @staticmethod
    def compute_episodic_return(batch, v_s_=None,
                                gamma=0.99, gae_lambda=0.95):
        """Compute returns over given full-length episodes, including the
        implementation of Generalized Advantage Estimation (arXiv:1506.02438).

        :param batch: a data batch which contains several full-episode data
            chronologically.
        :type batch: :class:`~tianshou.data.Batch`
        :param v_s_: the value function of all next states :math:`V(s')`.
        :type v_s_: numpy.ndarray
        :param float gamma: the discount factor, should be in [0, 1], defaults
            to 0.99.
        :param float gae_lambda: the parameter for Generalized Advantage
            Estimation, should be in [0, 1], defaults to 0.95.
        """
        if v_s_ is None:
            v_s_ = np.zeros_like(batch.rew)
        else:
            if not isinstance(v_s_, np.ndarray):
                v_s_ = np.array(v_s_, np.float)
            v_s_ = v_s_.reshape(batch.rew.shape)
        batch.returns = np.roll(v_s_, 1, axis=0)
        m = (1. - batch.done) * gamma
        delta = batch.rew + v_s_ * m - batch.returns
        m *= gae_lambda
        gae = 0.
        for i in range(len(batch.rew) - 1, -1, -1):
            gae = delta[i] + m[i] * gae
            batch.returns[i] += gae
        return batch
