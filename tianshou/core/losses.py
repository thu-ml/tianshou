import tensorflow as tf


def ppo_clip(sampled_action, advantage, clip_param, pi, pi_old):
    """
    the clip loss in ppo paper

    :param sampled_action: placeholder of sampled actions during interaction with the environment
    :param advantage: placeholder of estimated advantage values.
    :param clip param: float or Tensor of type float.
    :param pi: current `policy` to be optimized
    :param pi_old: old `policy` for computing the ppo loss as in Eqn. (7) in the paper
    """

    log_pi_act = pi.log_prob(sampled_action)
    log_pi_old_act = pi_old.log_prob(sampled_action)
    ratio = tf.exp(log_pi_act - log_pi_old_act)
    clipped_ratio = tf.clip_by_value(ratio, 1. - clip_param, 1. + clip_param)
    ppo_clip_loss = -tf.reduce_mean(tf.minimum(ratio * advantage, clipped_ratio * advantage))
    return ppo_clip_loss


def vanilla_policy_gradient(sampled_action, reward, pi, baseline="None"):
    """
    vanilla policy gradient

    :param sampled_action: placeholder of sampled actions during interaction with the environment
    :param reward: placeholder of reward the 'sampled_action' get
    :param pi: current 'policy' to be optimized
    :param baseline: the baseline method used to reduce the variance, default is 'None'
    :return:
    """
    log_pi_act = pi.log_prob(sampled_action)
    vanilla_policy_gradient_loss = tf.reduce_mean(reward * log_pi_act)
    # TODO： Different baseline methods like REINFORCE, etc.
    return vanilla_policy_gradient_loss

def temporal_difference_loss():
    pass

def deterministic_policy_gradient():
    pass