from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from tianshou.highlevel.persistence import (
    Persistence,
    PersistEvent,
    PolicyPersistence,
    RestoreEvent,
)
from tianshou.highlevel.world import World


@pytest.mark.parametrize("mode", list(PolicyPersistence.Mode))
@pytest.mark.parametrize("with_additional_persistence", [False, True])
def test_policy_persistence_round_trip(
    tmp_path: Path,
    mode: PolicyPersistence.Mode,
    with_additional_persistence: bool,
) -> None:
    policy = torch.nn.Linear(2, 1)
    with torch.no_grad():
        policy.weight.copy_(torch.tensor([[2.0, 3.0]]))
        policy.bias.fill_(4.0)
    inputs = torch.tensor([[1.0, 2.0]])
    expected_output = policy(inputs).detach().clone()
    expected_state = {name: value.clone() for name, value in policy.state_dict().items()}
    world = World(
        envs=Mock(),
        algorithm=Mock(),
        logger=Mock(),
        persist_directory=str(tmp_path),
        restore_directory=str(tmp_path),
    )
    additional_persistence = Mock(spec=Persistence) if with_additional_persistence else None
    persistence = PolicyPersistence(additional_persistence=additional_persistence, mode=mode)

    persistence.persist(policy, world)
    assert (tmp_path / mode.get_filename()).is_file()
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
    assert not torch.equal(policy(inputs), expected_output)

    persistence.restore(policy, world, device="cpu")

    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, expected_state[name])
        assert value.device.type == "cpu"
    torch.testing.assert_close(policy(inputs), expected_output)
    if additional_persistence is not None:
        additional_persistence.persist.assert_called_once_with(PersistEvent.PERSIST_POLICY, world)
        additional_persistence.restore.assert_called_once_with(RestoreEvent.RESTORE_POLICY, world)
