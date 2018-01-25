import tensorflow as tf


def DPG(policy, action_value):
    """
    construct the gradient tensor of deterministic policy gradient
    :param policy:
    :param action_value:
    :return: list of (gradient, variable) pairs
    """
    trainable_variables = policy.trainable_variables
    critic_action_input = action_value._action_placeholder
    critic_value_loss = -tf.reduce_mean(action_value.value_tensor)
    policy_action_output = policy.action

    grad_ys = tf.gradients(critic_value_loss, critic_action_input)
    grad_policy_vars = tf.gradients(policy_action_output, trainable_variables, grad_ys=grad_ys)
    # TODO: this is slightly different from ddpg implementations in baselines, keras-rl and rllab. it uses sampled action (with noise) rather than directly connect the two networks

    grads_and_vars = zip(grad_policy_vars, trainable_variables)

    return grads_and_vars