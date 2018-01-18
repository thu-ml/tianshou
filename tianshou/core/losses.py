import tensorflow as tf


def ppo_clip(policy, clip_param):
    """
    the clip loss in ppo paper

    :param sampled_action: placeholder of sampled actions during interaction with the environment
    :param advantage: placeholder of estimated advantage values.
    :param clip param: float or Tensor of type float.
    :param policy: current `policy` to be optimized
    :param pi_old: old `policy` for computing the ppo loss as in Eqn. (7) in the paper
    """
    action_ph = tf.placeholder(policy.act_dtype, shape=(None,) + policy.action_shape, name='ppo_clip_loss/action_placeholder')
    advantage_ph = tf.placeholder(tf.float32, shape=(None,), name='ppo_clip_loss/advantage_placeholder')
    policy.managed_placeholders['action'] = action_ph
    policy.managed_placeholders['advantage'] = advantage_ph

    log_pi_act = policy.log_prob(action_ph)
    log_pi_old_act = policy.log_prob_old(action_ph)
    ratio = tf.exp(log_pi_act - log_pi_old_act)
    clipped_ratio = tf.clip_by_value(ratio, 1. - clip_param, 1. + clip_param)
    ppo_clip_loss = -tf.reduce_mean(tf.minimum(ratio * advantage_ph, clipped_ratio * advantage_ph))
    return ppo_clip_loss


def REINFORCE(policy):
    """
    vanilla policy gradient

    :param sampled_action: placeholder of sampled actions during interaction with the environment
    :param reward: placeholder of reward the 'sampled_action' get
    :param pi: current `policy` to be optimized
    :param baseline: the baseline method used to reduce the variance, default is 'None'
    :return:
    """
    action_ph = tf.placeholder(policy.act_dtype, shape=(None,) + policy.action_shape,
                               name='REINFORCE/action_placeholder')
    advantage_ph = tf.placeholder(tf.float32, shape=(None,), name='REINFORCE/advantage_placeholder')
    policy.managed_placeholders['action'] = action_ph
    policy.managed_placeholders['advantage'] = advantage_ph

    log_pi_act = policy.log_prob(action_ph)
    REINFORCE_loss = -tf.reduce_mean(advantage_ph * log_pi_act)
    return REINFORCE_loss


def value_mse(state_value_function):
    """
    L2 loss of state value
    :param state_value_function: instance of StateValue
    :return: tensor of the mse loss
    """
    target_value_ph = tf.placeholder(tf.float32, shape=(None,), name='value_mse/return_placeholder')
    state_value_function.managed_placeholders['return'] = target_value_ph

    state_value = state_value_function.value_tensor
    return tf.losses.mean_squared_error(target_value_ph, state_value)


def qlearning(action_value_function):
    """
    deep q-network
    :param action_value_function: current `action_value` to be optimized
    :return:
    """
    target_value_ph = tf.placeholder(tf.float32, shape=(None,), name='qlearning/action_value_placeholder')
    action_value_function.managed_placeholders['return'] = target_value_ph

    q_value = action_value_function.value_tensor
    return tf.losses.mean_squared_error(target_value_ph, q_value)


def deterministic_policy_gradient(sampled_state, critic):
    """
    deterministic policy gradient:

    :param sampled_action: placeholder of sampled actions during the interaction with the environment
    :param critic: current `value` function
    :return:
    """
    return tf.reduce_mean(critic.get_value(sampled_state))