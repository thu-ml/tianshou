from __future__ import absolute_import
from __future__ import division

import tensorflow as tf


class PolicyBase(object):
    """
    base class for policy. only provides `act` method with exploration
    """
    def act(self, observation, my_feed_dict):
        raise NotImplementedError()

    def reset(self):
        """
        for temporal correlated random process exploration, as in DDPG
        :return:
        """
        pass
