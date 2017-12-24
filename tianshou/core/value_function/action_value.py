from base import ValueFunctionBase
import tensorflow as tf


class ActionValue(ValueFunctionBase):
    """
    class of action values Q(s, a).
    """
    def __init__(self, value_tensor, observation_placeholder, action_placeholder):
        self._action_placeholder = action_placeholder
        super(ActionValue, self).__init__(
            value_tensor=value_tensor,
            observation_placeholder=observation_placeholder
        )

    def get_value(self, observation, action):
        """

        :param observation: numpy array of observations, of shape (batchsize, observation_dim).
        :param action: numpy array of actions, of shape (batchsize, action_dim)
        # TODO: Atari discrete action should have dim 1. Super Mario may should have, say, dim 5, where each can be 0/1
        :return: numpy array of state values, of shape (batchsize, )
        # TODO: dealing with the last dim of 1 in V(s) and Q(s, a)
        """
        sess = tf.get_default_session()
        return sess.run(self.get_value_tensor(), feed_dict=
        {self._observation_placeholder: observation, self._action_placeholder:action})[:, 0]


class DQN(ActionValue):
    """
    class of the very DQN architecture. Instead of feeding s and a to the network to get a value, DQN feed s to the
    network and the last layer is Q(s, *) for all actions
    """
    def __init__(self, value_tensor, observation_placeholder, action_placeholder):
        """
        :param value_tensor: of shape (batchsize, num_actions)
        :param observation_placeholder: of shape (batchsize, observation_dim)
        :param action_placeholder: of shape (batchsize, )
        """
        self._value_tensor_all_actions = value_tensor
        canonical_value_tensor = value_tensor[action_placeholder]  # maybe a tf.map_fn. for now it's wrong

        super(DQN, self).__init__(value_tensor=canonical_value_tensor,
                                  observation_placeholder=observation_placeholder,
                                  action_placeholder=action_placeholder)

    def get_value_all_actions(self, observation):
        sess = tf.get_default_session()
        return sess.run(self._value_tensor_all_actions, feed_dict={self._observation_placeholder: observation})

    def get_value_tensor_all_actions(self):
        return self._value_tensor_all_actions