import os
import tempfile

import pytest
import torch
from torch import nn

from tianshou.highlevel.persistence import PolicyPersistence


class _StubWorld:
    """Minimal stand-in for tianshou.highlevel.world.World.

    PolicyPersistence only relies on the path-forming methods, so a stub avoids having to
    construct the full World (which requires environments, a logger, etc.).
    """

    def __init__(self, directory: str) -> None:
        self.directory = directory

    def persist_path(self, filename: str) -> str:
        return os.path.join(self.directory, filename)

    def restore_path(self, filename: str) -> str:
        return os.path.join(self.directory, filename)


class _DummyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@pytest.mark.parametrize(
    "mode", [PolicyPersistence.Mode.POLICY, PolicyPersistence.Mode.POLICY_STATE_DICT]
)
def test_policy_persistence_restore_roundtrip(mode: PolicyPersistence.Mode) -> None:
    """Saving then restoring a policy must reproduce the same weights.

    Regression test for issue #1297: with torch>=2.6, ``torch.load`` defaults to
    ``weights_only=True``, which fails to unpickle the whole-policy object persisted in
    ``Mode.POLICY``.
    """
    with tempfile.TemporaryDirectory() as tmp:
        world = _StubWorld(tmp)
        original = _DummyPolicy()
        original_weight = original.net.weight.detach().clone()

        persistence = PolicyPersistence(mode=mode)
        persistence.persist(original, world)  # type: ignore[arg-type]

        restored = _DummyPolicy()
        persistence.restore(restored, world, device="cpu")  # type: ignore[arg-type]

        assert torch.allclose(restored.net.weight.detach(), original_weight)
