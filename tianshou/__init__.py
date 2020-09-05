from tianshou import data, env, utils, policy, trainer, exploration

# pre-compile some common-type function-call to produce the correct benchmark
# result: https://github.com/thu-ml/tianshou/pull/193#discussion_r480536371
utils.pre_compile()


__version__ = '0.2.6'

__all__ = [
    'env',
    'data',
    'utils',
    'policy',
    'trainer',
    'exploration',
]
