from functions import *

from tianshou.data.replay_buffer.utils import get_replay_buffer


def test_rank_based():
    conf = {'size': 50,
            'learn_start': 10,
            'partition_num': 5,
            'total_step': 100,
            'batch_size': 4}
    experience = get_replay_buffer('rank_based', conf)

    # insert to experience
    print 'test insert experience'
    for i in range(1, 51):
        # tuple, like(state_t, a, r, state_t_1, t)
        to_insert = (i, 1, 1, i, 1)
        experience.add(to_insert)
    print experience.priority_queue
    print experience._experience[1]
    print experience._experience[2]
    print 'test replace'
    to_insert = (51, 1, 1, 51, 1)
    experience.add(to_insert)
    print experience.priority_queue
    print experience._experience[1]
    print experience._experience[2]

    # sample
    print 'test sample'
    global_step = {'global_step': 51}
    sample, w, e_id = experience.sample(global_step)
    print sample
    print w
    print e_id

    # update delta to priority
    print 'test update delta'
    delta = [v for v in range(1, 5)]
    experience.update_priority(e_id, delta)
    print experience.priority_queue
    sample, w, e_id = experience.sample(global_step)
    print sample
    print w
    print e_id

    # rebalance
    print 'test rebalance'
    experience.rebalance()
    print experience.priority_queue

def test_proportional():
    conf = {'size': 50,
            'alpha': 0.7,
            'batch_size': 4}
    experience = get_replay_buffer('proportional', conf)

    # insert to experience
    print 'test insert experience'
    for i in range(1, 51):
        # tuple, like(state_t, a, r, state_t_1, t)
        to_insert = (i, 1, 1, i, 1)
        experience.add(to_insert, i)
    print experience.tree
    print experience.tree.get_val(1)
    print experience.tree.get_val(2)
    print 'test replace'
    to_insert = (51, 1, 1, 51, 1)
    experience.add(to_insert, 51)
    print experience.tree
    print experience.tree.get_val(1)
    print experience.tree.get_val(2)

    # sample
    print 'test sample'
    beta = {'beta': 0.005}
    sample, w, e_id = experience.sample(beta)
    print sample
    print w
    print e_id

    # update delta to priority
    print 'test update delta'
    delta = [v for v in range(1, 5)]
    experience.update_priority(e_id, delta)
    print experience.tree
    sample, w, e_id = experience.sample(beta)
    print sample
    print w
    print e_id

def test_naive():
    conf = {'size': 50}
    experience = get_replay_buffer('naive', conf)

    # insert to experience
    print 'test insert experience'
    for i in range(1, 51):
        # tuple, like(state_t, a, r, state_t_1, t)
        to_insert = (i, 1, 1, i, 1)
        experience.add(to_insert)
    print experience.memory
    print 'test replace'
    to_insert = (51, 1, 1, 51, 1)
    experience.add(to_insert)
    print experience.memory

    # sample
    print 'test sample'
    batch_size = {'batch_size': 5}
    sample, w, e_id = experience.sample(batch_size)
    print sample
    print w
    print e_id

    # update delta to priority
    print 'test update delta'
    delta = [v for v in range(1, 5)]
    experience.update_priority(e_id, delta)
    print experience.memory
    sample, w, e_id = experience.sample(batch_size)
    print sample
    print w
    print e_id


if __name__ == '__main__':
    test_rank_based()
    test_proportional()
    test_naive()
