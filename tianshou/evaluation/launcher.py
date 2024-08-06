"""Provides a basic interface for launching experiments. The API is experimental and subject to change!."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from copy import copy
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Literal

from joblib import Parallel, delayed

from tianshou.data import InfoStats
from tianshou.highlevel.experiment import Experiment

log = logging.getLogger(__name__)


@dataclass
class JoblibConfig:
    n_jobs: int = -1
    """The maximum number of concurrently running jobs. If -1, all CPUs are used."""
    backend: Literal["loky", "multiprocessing", "threading"] | None = "loky"
    """Allows to hard-code backend, otherwise inferred based on prefer and require."""
    verbose: int = 10
    """If greater than zero, prints progress messages."""


class ExpLauncher(ABC):
    """Base interface for launching multiple experiments simultaneously."""

    def __init__(
        self,
        experiment_runner: Callable[
            [Experiment],
            InfoStats | None,
        ] = lambda exp: exp.run().trainer_result,
    ):
        """
        :param experiment_runner: determines how an experiment is to be executed.
            Overriding the default can be useful, e.g., for using high-level interfaces
            to set up an experiment (or an experiment collection) and tinkering with it prior to execution.
            This need often arises when prototyping with mechanisms that are not yet supported by
            the high-level interfaces.
            Allows arbitrary things to happen during experiment execution, so use it with caution!.
        """
        self.experiment_runner = experiment_runner

    @abstractmethod
    def _launch(self, experiments: Sequence[Experiment]) -> list[InfoStats | None]:
        """Should call `self.experiment_runner` for each experiment in experiments and aggregate the results."""

    def _safe_execute(self, exp: Experiment) -> InfoStats | None | Literal["failed"]:
        try:
            return self.experiment_runner(exp)
        except BaseException as e:
            log.error(f"Failed to run experiment {exp}.", exc_info=e)
            return "failed"

    @staticmethod
    def _return_from_successful_and_failed_exps(
        successful_exp_stats: list[InfoStats | None],
        failed_exps: list[Experiment],
    ) -> list[InfoStats | None]:
        if not successful_exp_stats:
            raise RuntimeError("All experiments failed, see error logs for more details.")
        if failed_exps:
            log.error(
                f"Failed to run the following "
                f"{len(failed_exps)}/{len(successful_exp_stats) + len(failed_exps)} experiments: {failed_exps}. "
                f"See the logs for more details. "
                f"Returning the results of {len(successful_exp_stats)} successful experiments.",
            )
        return successful_exp_stats

    def launch(self, experiments: Sequence[Experiment]) -> list[InfoStats | None]:
        """Will return the results of successfully executed experiments.

        If a single experiment is passed, will not use parallelism and run it in the main process.
        Failed experiments will be logged, and a RuntimeError is only raised if all experiments have failed.
        """
        if len(experiments) == 1:
            log.info(
                "A single experiment is being run, will not use parallelism and run it in the main process.",
            )
            return [self.experiment_runner(experiments[0])]
        return self._launch(experiments)


class SequentialExpLauncher(ExpLauncher):
    """Convenience wrapper around a simple for loop to run experiments sequentially."""

    def _launch(self, experiments: Sequence[Experiment]) -> list[InfoStats | None]:
        successful_exp_stats = []
        failed_exps = []
        for exp in experiments:
            exp_stats = self._safe_execute(exp)
            if exp_stats == "failed":
                failed_exps.append(exp)
            else:
                successful_exp_stats.append(exp_stats)
        # noinspection PyTypeChecker
        return self._return_from_successful_and_failed_exps(successful_exp_stats, failed_exps)


class JoblibExpLauncher(ExpLauncher):
    def __init__(
        self,
        joblib_cfg: JoblibConfig | None = None,
        experiment_runner: Callable[
            [Experiment],
            InfoStats | None,
        ] = lambda exp: exp.run().trainer_result,
    ) -> None:
        super().__init__(experiment_runner=experiment_runner)
        self.joblib_cfg = copy(joblib_cfg) if joblib_cfg is not None else JoblibConfig()
        # Joblib's backend is hard-coded to loky since the threading backend produces different results
        # TODO: fix this
        if self.joblib_cfg.backend != "loky":
            log.warning(
                f"Ignoring the user provided joblib backend {self.joblib_cfg.backend} and using loky instead. "
                f"The current implementation requires loky to work and will be relaxed soon",
            )
            self.joblib_cfg.backend = "loky"

    def _launch(self, experiments: Sequence[Experiment]) -> list[InfoStats | None]:
        results = Parallel(**asdict(self.joblib_cfg))(
            delayed(self._safe_execute)(exp) for exp in experiments
        )
        successful_exps = []
        failed_exps = []
        for exp, result in zip(experiments, results, strict=True):
            if result == "failed":
                failed_exps.append(exp)
            else:
                successful_exps.append(result)
        return self._return_from_successful_and_failed_exps(successful_exps, failed_exps)


class RegisteredExpLauncher(Enum):
    joblib = "joblib"
    sequential = "sequential"

    def create_launcher(self) -> ExpLauncher:
        match self:
            case RegisteredExpLauncher.joblib:
                return JoblibExpLauncher()
            case RegisteredExpLauncher.sequential:
                return SequentialExpLauncher()
            case _:
                raise NotImplementedError(
                    f"Launcher {self} is not yet implemented.",
                )
