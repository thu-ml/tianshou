from tianshou.utils.config import tqdm_config
from tianshou.utils.moving_average import MovAvg


def run_once(f):
    """
    Run once decorator for a method in a class. Each instance can run
    the method at most once.
    """
    f.has_run_objects = set()

    def wrapper(self, *args, **kwargs):
        if hash(self) in f.has_run_objects:
            raise RuntimeError(
                f'{f} can be called only once for object {self}')
        f.has_run_objects.add(hash(self))
        return f(self, *args, **kwargs)
    return wrapper


__all__ = [
    'MovAvg',
    'run_once',
    'tqdm_config',
]
