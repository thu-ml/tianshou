    #!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from six.moves import zip


tf.flags.DEFINE_integer('num_gpus', 1, """How many GPUs to use""")
tf.flags.DEFINE_boolean('log_device_placement', False,
                        """Whether to log device placement.""")
FLAGS = tf.flags.FLAGS


def create_session():
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True

    return tf.Session(config=config)


def average_gradients(tower_grads):
    """
    Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    :param tower_grads: List of lists of (gradient, variable) tuples.
        The outer list is over individual gradients. The inner list is over
        the gradient calculation for each tower.
    :return: List of pairs of (gradient, variable) where the gradient has
        been averaged across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        if grad_and_vars[0][0] is None:
            continue
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def average_losses(tower_losses):
    """
    Calculate the average loss or other quantity for all towers.

    :param tower_losses: A list of lists of quantities. The outer list is over
        towers. The inner list is over losses or other quantities for each
        tower.
    :return: A list of quantities that have been averaged over all towers.
    """
    ret = []
    for quantities in zip(*tower_losses):
        ret.append(tf.add_n(quantities) / len(quantities))
    return ret
