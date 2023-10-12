import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable, Callable

import torch

from tianshou.highlevel.world import World

if TYPE_CHECKING:
    from tianshou.highlevel.module.core import TDevice

log = logging.getLogger(__name__)


@runtime_checkable
class PersistableConfigProtocol(Protocol):
    @classmethod
    def load(cls, path: os.PathLike[str]) -> Self:
        pass

    def save(self, path: os.PathLike[str]) -> None:
        pass


class PersistEvent(Enum):
    PERSIST_POLICY = "persist_policy"
    """Policy neural network is persisted (new best found)"""


class RestoreEvent(Enum):
    RESTORE_POLICY = "restore_policy"
    """Policy neural network parameters are restored"""


class Persistence(ABC):
    @abstractmethod
    def persist(self, event: PersistEvent, world: World) -> None:
        pass

    @abstractmethod
    def restore(self, event: RestoreEvent, world: World):
        pass


class PersistenceGroup(Persistence):
    def __init__(self, *p: Persistence):
        self.items = p

    def persist(self, event: PersistEvent, world: World) -> None:
        for item in self.items:
            item.persist(event, world)

    def restore(self, event: RestoreEvent, world: World):
        for item in self.items:
            item.restore(event, world)


class PolicyPersistence:
    FILENAME = "policy.dat"

    def __init__(self, additional_persistence: Persistence | None = None):
        self.additional_persistence = additional_persistence

    def persist(self, policy: torch.nn.Module, world: World) -> None:
        path = world.persist_path(self.FILENAME)
        log.info(f"Saving policy in {path}")
        torch.save(policy.state_dict(), path)
        if self.additional_persistence is not None:
            self.additional_persistence.persist(PersistEvent.PERSIST_POLICY, world)

    def restore(self, policy: torch.nn.Module, world: World, device: "TDevice") -> None:
        path = world.restore_path(self.FILENAME)
        log.info(f"Restoring policy from {path}")
        state_dict = torch.load(path, map_location=device)
        policy.load_state_dict(state_dict)
        if self.additional_persistence is not None:
            self.additional_persistence.restore(RestoreEvent.RESTORE_POLICY, world)

    def get_save_best_fn(self, world) -> Callable[[torch.nn.Module], None]:
        def save_best_fn(pol: torch.nn.Module) -> None:
            self.persist(pol, world)

        return save_best_fn
