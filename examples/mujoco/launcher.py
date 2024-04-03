from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Literal

from joblib import Parallel, delayed

from tianshou.highlevel.experiment import Experiment


@dataclass
class JoblibConfig:
    n_jobs: int = -1
    """The maximum number of concurrently running jobs. If -1, all CPUs are used."""
    backend: Literal["loky", "multiprocessing", "threading"] | None = None
    """Allows to hard-code backend, otherwise inferred based on prefer and require."""
    verbose: int = 10
    """If greater than zero, prints progress messages."""


class ExpLauncher(ABC):
    @abstractmethod
    def launch(self, experiments: dict[str, Experiment]) -> None:
        raise NotImplementedError


class SequentialExpLauncher(ExpLauncher):
    def launch(self, experiments: dict[str, Experiment]) -> None:
        for exp_name, exp in experiments.items():
            exp.run(exp_name)


class JoblibExpLauncher(ExpLauncher):
    def __init__(self, joblib_cfg: JoblibConfig | None = None) -> None:
        self.joblib_cfg = joblib_cfg or JoblibConfig()
        # Joblib's backend is hard-coded to loky since the threading backend produces different results
        self.joblib_cfg.backend = "loky"

    def launch(self, experiments: dict[str, Experiment]) -> None:
        Parallel(**asdict(self.joblib_cfg))(
            delayed(self.execute_task)(exp, exp_name) for exp_name, exp in experiments.items()
        )

    @staticmethod
    def execute_task(exp: Experiment, name: str):
        try:
            exp.run(name)
        except Exception as e:
            print(e)


class RegisteredExpLauncher(Enum):
    joblib = "joblib"
    sequential = "sequential"

    def create_launcher(self):
        match self:
            case RegisteredExpLauncher.joblib:
                return JoblibExpLauncher()
            case RegisteredExpLauncher.sequential:
                return SequentialExpLauncher()
            case _:
                raise NotImplementedError(
                    f"Launcher {self} is not implemented, registered launchers are {list(RegisteredExpLauncher)}.",
                )
