import tensorflow as tf


def identify_dependent_variables(tensor, candidate_variables):
    """
    identify the variables that `tensor` depends on
    :param tensor: A Tensor.
    :param candidate_variables: A list of Variables.
    :return: A list of variables in `candidate variables` that has effect on `tensor`
    """
    grads = tf.gradients(tensor, candidate_variables)
    return [var for var, grad in zip(candidate_variables, grads) if grad is not None]


def get_soft_update_op(update_fraction, including_nets, excluding_nets=None):
    """

    :param including_nets:
    :param excluding_nets:
    :return:
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
