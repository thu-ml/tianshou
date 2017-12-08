import tensorflow as tf
import baselines.common.tf_util as U


def ppo_clip(sampled_action, Dgrad, clip_param, pi, pi_old):
    log_pi_act = pi.log_prob(sampled_action)
    log_pi_old_act = pi_old.log_prob(sampled_action)
    ratio = tf.exp(log_pi_act - log_pi_old_act)
    clipped_ratio = tf.clip_by_value(ratio, 1. - clip_param, 1. + clip_param)
    ppo_clip_loss = -tf.reduce_mean(tf.minimum(ratio * Dgrad, clipped_ratio * Dgrad))
    return ppo_clip_loss


def L_VF(Gt, pi, St): # TODO: do we really have to specify St, or it's implicit in policy/value net
    return U.mean(tf.square(pi.vpred - Gt))


def entropy_reg(pi):
    return - U.mean(pi.pd.entropy())


def KL_diff(pi, pi_old):
    kloldnew = pi_old.pd.kl(pi.pd)
    meankl = U.mean(kloldnew)
    return meankl