import sys

from tianshou.data.replay_buffer.naive import NaiveExperience
from tianshou.data.replay_buffer.proportional import PropotionalExperience
from tianshou.data.replay_buffer.rank_based import RankBasedExperience


def get_replay_buffer(name, env, policy, qnet, target_qnet, conf):
    """
    Get replay buffer according to the given name.
    """

    if name == 'rank_based':
        return RankBasedExperience(env, policy, qnet, target_qnet, conf)
    elif name == 'proportional':
        return PropotionalExperience(env, policy, qnet, target_qnet, conf)
    elif name == 'naive':
        return NaiveExperience(env, policy, qnet, target_qnet, conf)
    else:
        sys.stderr.write('no such replay buffer')
