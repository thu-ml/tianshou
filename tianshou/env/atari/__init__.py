"""Atari environment helpers for Tianshou.

Network classes (C51Net, DQNet, etc.) are always importable.
Wrapper classes (WarpFrame, make_atari_env, etc.) require optional
dependencies (cv2/gymnasium[atari]) and are imported lazily to avoid
breaking environments that only need the network utilities.
"""

from tianshou.env.atari.atari_network import (
    ActorFactoryAtariDQN,
    C51Net,
    DQNet,
    IntermediateModuleFactoryAtariDQN,
    IntermediateModuleFactoryAtariDQNFeatures,
    QRDQNet,
    RainbowNet,
    ScaledObsInputActionReprNet,
    layer_init,
)

# Wrapper symbols that require cv2 — imported lazily via __getattr__
_WRAPPER_SYMBOLS = {
    "AtariEnvFactory",
    "AtariEpochStopCallback",
    "ClipRewardEnv",
    "EpisodicLifeEnv",
    "FireResetEnv",
    "FrameStack",
    "MaxAndSkipEnv",
    "NoopResetEnv",
    "ScaledFloatFrame",
    "WarpFrame",
    "make_atari_env",
    "wrap_deepmind",
}

__all__ = [
    "ActorFactoryAtariDQN",
    "AtariEnvFactory",
    "AtariEpochStopCallback",
    "C51Net",
    "ClipRewardEnv",
    "DQNet",
    "EpisodicLifeEnv",
    "FireResetEnv",
    "FrameStack",
    "IntermediateModuleFactoryAtariDQN",
    "IntermediateModuleFactoryAtariDQNFeatures",
    "MaxAndSkipEnv",
    "NoopResetEnv",
    "QRDQNet",
    "RainbowNet",
    "ScaledFloatFrame",
    "ScaledObsInputActionReprNet",
    "WarpFrame",
    "layer_init",
    "make_atari_env",
    "wrap_deepmind",
]


def __getattr__(name: str):
    if name in _WRAPPER_SYMBOLS:
        from tianshou.env.atari import atari_wrapper

        return getattr(atari_wrapper, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
