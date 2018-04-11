import tensorflow as tf


def DPG(policy, action_value):
    """
    construct the gradient tensor of deterministic policy gradient
    :param policy:
    :param action_value:
    :return: list of (gradient, variable) pairs
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