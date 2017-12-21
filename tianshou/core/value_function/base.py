
# TODO: linear feature baseline also in tf?
class ValueFunctionBase(object):
    """
    base class of value functions. Children include state values V(s) and action values Q(s, a)
    """
    def __init__(self, value_tensor, observation_placeholder):
        self._observation_placeholder = observation_placeholder
        self._value_tensor = value_tensor

    def get_value(self, **kwargs):
        """

        :return: batch of corresponding values in numpy array
        """
        raise NotImplementedError()

    def get_value_tensor(self):
        """

        :return: tensor of the corresponding values
        """
        return self._value_tensor
