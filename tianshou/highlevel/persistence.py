import os
from typing import Protocol, Self, runtime_checkable


@runtime_checkable
class PersistableConfigProtocol(Protocol):
    @classmethod
    def load(cls, path: os.PathLike[str]) -> Self:
        pass

    def save(self, path: os.PathLike[str]) -> None:
        pass
