import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING

import torch

from tianshou.highlevel.world import World

if TYPE_CHECKING:
    from tianshou.highlevel.module.core import TDevice

log = logging.getLogger(__name__)


class PersistEvent(Enum):
    """Enumeration of persistence events that Persistence objects can react to."""

    PERSIST_POLICY = "persist_policy"
    """Policy neural network is persisted (new best found)"""


class RestoreEvent(Enum):
    """Enumeration of restoration events that Persistence objects can react to."""

    RESTORE_POLICY = "restore_policy"
    """Policy neural network parameters are restored"""


class Persistence(ABC):
    @abstractmethod
    def persist(self, event: PersistEvent, world: World) -> None:
        pass

    @abstractmethod
    def restore(self, event: RestoreEvent, world: World) -> None:
        pass


class PersistenceGroup(Persistence):
    """Groups persistence handler such that they can be applied collectively."""

    def __init__(self, *p: Persistence, enabled: bool = True):
        self.items = p
        self.enabled = enabled

    def persist(self, event: PersistEvent, world: World) -> None:
        if not self.enabled:
            return
        for item in self.items:
            item.persist(event, world)

    def restore(self, event: RestoreEvent, world: World) -> None:
        for item in self.items:
            item.restore(event, world)


class PolicyPersistence:
    class Mode(Enum):
        """Mode of persistence."""

        POLICY_STATE_DICT = "policy_state_dict"
        """Persist only the policy's state dictionary. Note that for a policy to be restored from
        such a dictionary, it is necessary to first create a structurally equivalent object which can
        accept the respective state."""
        POLICY = "policy"
        """Persist the entire policy. This is larger but has the advantage of the policy being loadable
        without requiring an environment to be instantiated.
        It has the potential disadvantage that upon breaking code changes in the policy implementation
        (e.g. renamed/moved class), it will no longer be loadable.
        Note that a precondition is that the policy be picklable in its entirety.
        """

        def get_filename(self) -> str:
            return self.value + ".pt"

    def __init__(
        self,
        additional_persistence: Persistence | None = None,
        enabled: bool = True,
        mode: Mode = Mode.POLICY,
    ):
        """Handles persistence of the policy.

        :param additional_persistence: a persistence instance which is to be invoked whenever
            this object is used to persist/restore data
        :param enabled: whether persistence is enabled (restoration is always enabled)
        :param mode: the persistence mode
        """
        self.additional_persistence = additional_persistence
        self.enabled = enabled
        self.mode = mode

    def persist(self, policy: torch.nn.Module, world: World) -> None:
        if not self.enabled:
            return
        path = world.persist_path(self.mode.get_filename())
        match self.mode:
            case self.Mode.POLICY_STATE_DICT:
                log.info(f"Saving policy state dictionary in {path}")
                torch.save(policy.state_dict(), path)
            case self.Mode.POLICY:
                log.info(f"Saving policy object in {path}")
                torch.save(policy, path)
            case _:
                raise NotImplementedError
        if self.additional_persistence is not None:
            self.additional_persistence.persist(PersistEvent.PERSIST_POLICY, world)

    def restore(self, policy: torch.nn.Module, world: World, device: "TDevice") -> None:
        path = world.restore_path(self.mode.get_filename())
        log.info(f"Restoring policy from {path}")
        match self.mode:
            case self.Mode.POLICY_STATE_DICT:
                state_dict = torch.load(path, map_location=device)
            case self.Mode.POLICY:
                loaded_policy: torch.nn.Module = torch.load(path, map_location=device)
                state_dict = loaded_policy.state_dict()
            case _:
                raise NotImplementedError
        policy.load_state_dict(state_dict)
        if self.additional_persistence is not None:
            self.additional_persistence.restore(RestoreEvent.RESTORE_POLICY, world)

    def get_save_best_fn(self, world: World) -> Callable[[torch.nn.Module], None]:
        def save_best_fn(pol: torch.nn.Module) -> None:
            self.persist(pol, world)

        return save_best_fn
