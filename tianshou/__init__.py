import os

from tianshou import data, env, utils, policy, trainer, \
    exploration

version_file = os.path.join(os.path.dirname(__file__), "version.txt")

__version__ = open(version_file).read().strip()
__all__ = [
    'env',
    'data',
    'utils',
    'policy',
    'trainer',
    'exploration',
]
