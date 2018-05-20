import tensorflow as tf

__all__ = [
    'DPG',
]


def DPG(policy, action_value):
    """
    Constructs the gradient Tensor of `deterministic policy gradient <https://arxiv.org/pdf/1509.02971.pdf>`_.

    :param policy: A :class:`tianshou.core.policy.Deterministic` to be optimized.
    :param action_value: A :class:`tianshou.core.value_function.ActionValue` to guide the optimization of `policy`.

    :return: A list of (gradient, variable) pairs.
    """
    trainable_variables = list(policy.trainable_variables)
    critic_action_input = action_value.action_placeholder
    critic_value_loss = -tf.reduce_mean(action_value.value_tensor)
    policy_action_output = policy.action

    grad_ys = tf.gradients(critic_value_loss, critic_action_input)[0]
    # stop gradient in case policy and action value have shared variables
    grad_ys = tf.stop_gradient(grad_ys)

    deterministic_policy_grads = tf.gradients(policy_action_output, trainable_variables, grad_ys=grad_ys)

    grads_and_vars = [(grad, var) for grad, var in zip(deterministic_policy_grads, trainable_variables)]

    return grads_and_vars