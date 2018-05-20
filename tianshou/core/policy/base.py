from __future__ import absolute_import
from __future__ import division

__all__ = []


class PolicyBase(object):
    """
    Base class for policy. Mandatory methods for a policy class are:

    - :func:`act`. It's used interacting with the environment during training, \
    so exploration noise should be added in this method.

    - :func:`act_test`. Since RL usually adds additional exploration noise during training, a different method\
    for testing the policy should be defined with different exploration specification.\
    Generally, DQN uses different :math:`\epsilon` in :math:`\epsilon`-greedy and\
    DDPG removes exploration noise during test.

    - :func:`reset`. It's mainly to reset the states of the exploration random process, or if your policy has\
    some internal states that should be reset at the beginning of each new episode. Otherwise, this method\
    does nothing.
    """
    def act(self, observation, my_feed_dict):
        """
        Return action given observation, when interacting with the environment during training.

        :param observation: An array-like with rank the same as a single observation of the environment.
            Its "batch_size" is 1, but should not be explicitly set. This method will add the dimension
            of "batch_size" to the first dimension.
        :param my_feed_dict: A dict. Specifies placeholders such as dropout and batch_norm except observation.

        :return: A numpy array. Action given the single observation. Its "batch_size" is 1,
            but should not be explicitly set.
        """
        raise NotImplementedError()

    def act_test(self, observation, my_feed_dict):
        """
        Return action given observation, when interacting with the environment during test.

        :param observation: An array-like with rank the same as a single observation of the environment.
            Its "batch_size" is 1, but should not be explicitly set. This method will add the dimension
            of "batch_size" to the first dimension.
        :param my_feed_dict: A dict. Specifies placeholders such as dropout and batch_norm except observation.

        :return: A numpy array. Action given the single observation. Its "batch_size" is 1,
            but should not be explicitly set.
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset the internal states of the policy. Does nothing by default.
        """
        pass
