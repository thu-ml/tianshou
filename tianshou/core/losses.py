import tensorflow as tf


def ppo_clip(policy, clip_param):
    """
    Builds the graph of clipped loss :math:`L^{CLIP}` as in the
    `PPO paper <https://arxiv.org/pdf/1707.06347.pdf>`_, which is basically
    :math:`-\min(r_t(\\theta)A_t, \mathrm{clip}(r_t(\\theta), 1 - \epsilon, 1 + \epsilon)A_t)`.
    We minimize the objective instead of maximizing, hence the leading negative sign.
    It creates an action placeholder and an advantage placeholder and adds into the ``managed_placeholders``
    of the ``policy``.

    :param policy: A :class:`tianshou.core.policy` to be optimized.
    :param clip param: A float or Tensor of type float. The :math:`\epsilon` in the loss equation.

    :return: A scalar float Tensor of the loss.
    """
    action_ph = tf.placeholder(policy.action.dtype, shape=policy.action.shape, name='ppo_clip_loss/action_placeholder')
    advantage_ph = tf.placeholder(tf.float32, shape=(None,), name='ppo_clip_loss/advantage_placeholder')
    policy.managed_placeholders['action'] = action_ph
    policy.managed_placeholders['advantage'] = advantage_ph

    log_pi_act = policy.action_dist.log_prob(action_ph)
    log_pi_old_act = policy.action_dist_old.log_prob(action_ph)
    ratio = tf.exp(log_pi_act - log_pi_old_act)
    clipped_ratio = tf.clip_by_value(ratio, 1. - clip_param, 1. + clip_param)
    ppo_clip_loss = -tf.reduce_mean(tf.minimum(ratio * advantage_ph, clipped_ratio * advantage_ph))
    return ppo_clip_loss


def REINFORCE(policy):
    """
    Builds the graph of the loss function as used in vanilla policy gradient algorithms, i.e., REINFORCE.
    The loss is basically :math:`\log \pi(a|s) A_t`.
    We minimize the objective instead of maximizing, hence the leading negative sign.
    It creates an action placeholder and an advantage placeholder and adds into the ``managed_placeholders``
    of the ``policy``.

    :param policy: A :class:`tianshou.core.policy` to be optimized.

    :return: A scalar float Tensor of the loss.
    """
    action_ph = tf.placeholder(policy.action.dtype, shape=policy.action.shape,
                               name='REINFORCE/action_placeholder')
    advantage_ph = tf.placeholder(tf.float32, shape=(None,), name='REINFORCE/advantage_placeholder')
    policy.managed_placeholders['action'] = action_ph
    policy.managed_placeholders['advantage'] = advantage_ph

    log_pi_act = policy.action_dist.log_prob(action_ph)
    REINFORCE_loss = -tf.reduce_mean(advantage_ph * log_pi_act)
    return REINFORCE_loss


def value_mse(value_function):
    """
    Builds the graph of L2 loss on value functions for, e.g., training critics or DQN.
    It creates an placeholder for the target value adds it into the ``managed_placeholders``
    of the ``value_function``.

    :param value_function: A :class:`tianshou.core.value_function` to be optimized.

    :return: A scalar float Tensor of the loss.
    """
    target_value_ph = tf.placeholder(tf.float32, shape=(None,), name='value_mse/return_placeholder')
    value_function.managed_placeholders['return'] = target_value_ph

    state_value = value_function.value_tensor
    return tf.losses.mean_squared_error(target_value_ph, state_value)
