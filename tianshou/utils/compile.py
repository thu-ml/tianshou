import numpy as np

# functions that need to pre-compile for producing benchmark result
from tianshou.policy.base import _episodic_return, _nstep_return
from tianshou.data.utils.segtree import _reduce, _setitem, _get_prefix_sum_idx


def pre_compile():
    """Since Numba acceleration needs to compile the function in the first run,
    here we use some fake data for the common-type function-call compilation.
    Otherwise, the current training speed cannot compare with the previous.
    """
    f64 = np.array([0, 1], dtype=np.float64)
    f32 = np.array([0, 1], dtype=np.float32)
    b = np.array([False, True], dtype=np.bool_)
    i64 = np.array([0, 1], dtype=np.int64)
    # returns
    _episodic_return(f64, f64, b, .1, .1)
    _episodic_return(f32, f64, b, .1, .1)
    _nstep_return(f64, b, f32, i64, .1, 1, 4, 1., 0.)
    # segtree
    _setitem(f64, i64, f64)
    _setitem(f64, i64, f32)
    _reduce(f64, 0, 1)
    _get_prefix_sum_idx(f64, 1, f64)
    _get_prefix_sum_idx(f32, 1, f64)
