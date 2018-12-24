import tensorflow as tf
from .base import PolicyBase
from ..utils import identify_dependent_variables

__all__ = [
    'Distributional',
]


class Distributional(PolicyBase):
    """
    Policy class where action is specified by a probability distribution. Depending on the distribution,
    it can be applied to both continuous and discrete action spaces.

    :param network_callable: A Python callable returning (action head, value head). When called it builds the tf graph and returns a
        :class:`tf.distributions.Distribution` on the action space on the action head.
    :param observation_placeholder: A :class:`tf.placeholder`. The observation placeholder of the network graph.
    :param has_old_net: A bool defaulting to ``False``. If true this class will create another graph with another
        set of :class:`tf.Variable` s to be the "old net". The "old net" could be the target networks as in DQN
        and DDPG, or just an old net to help optimization as in PPO.
    """
    def __init__(self, network_callable, observation_placeholder, has_old_net=False):
        self.observation_placeholder = observation_placeholder
        self.managed_placeholders = {'observation': observation_placeholder}

        self.has_old_net = has_old_net

        network_scope = 'network'
        net_old_scope = 'net_old'

        # build network, action and value
        with tf.variable_scope(network_scope, reuse=tf.AUTO_REUSE):
            action_dist = network_callable()[0]
            assert action_dist is not None
            self.action_dist = action_dist
            self.action = action_dist.sample()

        weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.network_weights = identify_dependent_variables(self.action_dist.log_prob(tf.stop_gradient(self.action)), weights)
        self._trainable_variables = [var for var in self.network_weights
                                    if var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
        # deal with target network
        if not has_old_net:
            self.sync_weights_ops = None
        else:  # then we need to build another tf graph as target network
            with tf.variable_scope('net_old', reuse=tf.AUTO_REUSE):
                self.action_dist_old = network_callable()[0]
                self.action_old = self.action_dist_old.sample()

            old_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=net_old_scope)

            # re-filter to rule out some edge cases
            old_weights = [var for var in old_weights if var.name[:len(net_old_scope)] == net_old_scope]

            self.network_old_weights = identify_dependent_variables(self.action_dist_old.log_prob(tf.stop_gradient(self.action_old)), old_weights)
            assert len(self.network_weights) == len(self.network_old_weights)

            self.sync_weights_ops = [tf.assign(variable_old, variable)
                                     for (variable_old, variable) in
                                     zip(self.network_old_weights, self.network_weights)]

    @property
    def trainable_variables(self):
        """
        The trainable variables of the policy in a Python **set**. It contains only the :class:`tf.Variable` s
        that affect the action.
        """
        return set(self._trainable_variables)

    def act(self, observation, my_feed_dict={}):
        """
        Return action given observation, directly sampling from the action distribution.

        :param observation: An array-like with rank the same as a single observation of the environment.
            Its "batch_size" is 1, but should not be explicitly set. This method will add the dimension
            of "batch_size" to the first dimension.
        :param my_feed_dict: Optional. A dict defaulting to empty.
            Specifies placeholders such as dropout and batch_norm except observation.

        :return: A numpy array.
            Action given the single observation. Its "batch_size" is 1, but should not be explicitly set.
        """
        sess = tf.get_default_session()

        # observation[None] adds one dimension at the beginning
        feed_dict = {self.observation_placeholder: observation[None]}
        feed_dict.update(my_feed_dict)
        sampled_action = sess.run(self.action, feed_dict=feed_dict)
        sampled_action = sampled_action[0]

        return sampled_action

    def act_test(self, observation, my_feed_dict={}):
        """
        Return action given observation, directly sampling from the action distribution.

        :param observation: An array-like with rank the same as a single observation of the environment.
            Its "batch_size" is 1, but should not be explicitly set. This method will add the dimension
            of "batch_size" to the first dimension.
        :param my_feed_dict: Optional. A dict defaulting to empty.
            Specifies placeholders such as dropout and batch_norm except observation.

        :return: A numpy array.
            Action given the single observation. Its "batch_size" is 1, but should not be explicitly set.
        """
        return self.act(observation, my_feed_dict)

    def sync_weights(self):
        """
        Sync the variables of the "old net" to be the same as the current network.
        """
        if self.sync_weights_ops is not None:
            sess = tf.get_default_session()
            sess.run(self.sync_weights_ops)
