import tensorflow as tf

__all__ = [
    'get_soft_update_op',
]


def identify_dependent_variables(tensor, candidate_variables):
    """
    Identify and return the variables in ``candidate_variables`` that ``tensor`` depends on.

    :param tensor: A Tensor. The target Tensor to identify dependency.
    :param candidate_variables: A list of :class:`tf.Variable` s. The candidate Variables to identify dependency.

    :return: A list of :class:`tf.Variable` s in ``candidate variables`` that has effect on ``tensor``.
    """
    grads = tf.gradients(tensor, candidate_variables)
    return [var for var, grad in zip(candidate_variables, grads) if grad is not None]


def get_soft_update_op(update_fraction, including_nets, excluding_nets=None):
    """
    Builds the graph op to softly update the "old net" of policies and value_functions, as suggested in
    `DDPG <https://arxiv.org/pdf/1509.02971.pdf>`_. It updates the :class:`tf.Variable` s in the old net,
    :math:`\\theta'` with the :class:`tf.Variable` s in the current network, :math:`\\theta` as
    :math:`\\theta' = \\tau \\theta + (1 - \\tau) \\theta'`.

    :param update_fraction: A float in range :math:`[0, 1]`. Corresponding to the :math:`\\tau` in the update equation.
    :param including_nets: A list of policies and/or value_functions. All :class:`tf.Variable` s in these networks
        are included for update. Shared Variables are updated only once in case of layer sharing among the networks.
    :param excluding_nets: Optional. A list of policies and/or value_functions defaulting to ``None``.
        All :class:`tf.Variable` s in these networks
        are excluded from the update determined by ``including nets``. This is useful in existence of layer sharing
        among networks and we only want to update the Variables in ``including_nets`` that are not shared.

    :return: A list of ops :func:`tf.assign` specifying the soft update.
    """
    assert 0 < update_fraction < 1, 'Unrecommended update_fraction <=0 or >=1!'

    variables = []
    variables_old = []
    for net in including_nets:
        for var, var_old in zip(net.network_weights, net.network_old_weights):
            if var not in variables:
                variables.append(var)
                variables_old.append(var_old)

    if excluding_nets:
        excluding_variables = []
        for net in excluding_nets:
            excluding_variables += net.network_weights
        for var, var_old in zip(variables, variables_old):
            if var in excluding_variables:
                variables.remove(var)
                variables_old.remove(var_old)

    return [tf.assign(var_old, update_fraction * var + (1 - update_fraction) * var_old)
            for var_old, var in zip(variables_old, variables)]
