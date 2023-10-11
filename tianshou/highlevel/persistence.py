import logging
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

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


class Persistence(ABC):
    def path(self, world: World, filename: str) -> str:
        return os.path.join(world.directory, filename)

    @abstractmethod
    def persist(self, world: World) -> None:
        pass

    @abstractmethod
    def restore(self, world: World):
        pass


class PersistenceGroup(Persistence):
    def __init__(self, *p: Persistence):
        self.items = p

    def persist(self, world: World) -> None:
        for item in self.items:
            item.persist(world)

    def restore(self, world: World):
        for item in self.items:
            item.restore(world)


class PolicyPersistence:
    FILENAME = "policy.dat"

    def persist(self, policy: torch.nn.Module, directory: str) -> None:
        path = os.path.join(directory, self.FILENAME)
        log.info(f"Saving policy in {path}")
        torch.save(policy.state_dict(), path)

    def restore(self, policy: torch.nn.Module, directory: str, device: "TDevice") -> None:
        path = os.path.join(directory, self.FILENAME)
        log.info(f"Restoring policy from {path}")
        state_dict = torch.load(path, map_location=device)
        policy.load_state_dict(state_dict)

    def get_save_best_fn(self, world):
        def save_best_fn(pol: torch.nn.Module) -> None:
            self.persist(pol, world.directory)

        return save_best_fn
