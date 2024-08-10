"""The rliable-evaluation module provides a high-level interface to evaluate the results of an experiment with multiple runs
on different seeds using the rliable library. The API is experimental and subject to change!.
"""

import os
from dataclasses import dataclass, fields
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sst
from rliable import library as rly
from rliable import plot_utils
from sensai.util import logging

from tianshou.highlevel.experiment import Experiment
from tianshou.utils import TensorboardLogger
from tianshou.utils.logger.base import DataScope

log = logging.getLogger(__name__)


@dataclass
class LoggedSummaryData:
    mean: np.ndarray
    std: np.ndarray
    max: np.ndarray
    min: np.ndarray


@dataclass
class LoggedCollectStats:
    env_step: np.ndarray | None = None
    n_collected_episodes: np.ndarray | None = None
    n_collected_steps: np.ndarray | None = None
    collect_time: np.ndarray | None = None
    collect_speed: np.ndarray | None = None
    returns_stat: LoggedSummaryData | None = None
    lens_stat: LoggedSummaryData | None = None

    @classmethod
    def from_data_dict(cls, data: dict) -> "LoggedCollectStats":
        """Create a LoggedCollectStats object from a dictionary.

        Converts SequenceSummaryStats from dict format to dataclass format and ignores fields that are not present.
        """
        field_names = [f.name for f in fields(cls)]
        for k, v in data.items():
            if k not in field_names:
                data.pop(k)
            if isinstance(v, dict):
                data[k] = LoggedSummaryData(**v)
        return cls(**data)


@dataclass
class RLiableExperimentResult:
    """The result of an experiment that can be used with the rliable library."""

    exp_dir: str
    """The base directory where each sub-directory contains the results of one experiment run."""

    test_episode_returns_RE: np.ndarray
    """The test episodes for each run of the experiment where each row corresponds to one run."""

    train_episode_returns_RE: np.ndarray
    """The training episodes for each run of the experiment where each row corresponds to one run."""

    env_steps_E: np.ndarray
    """The number of environment steps at which the test episodes were evaluated."""

    env_steps_train_E: np.ndarray
    """The number of environment steps at which the training episodes were evaluated."""

    @classmethod
    def load_from_disk(
        cls,
        exp_dir: str,
        max_env_step: int | None = None,
    ) -> "RLiableExperimentResult":
        """Load the experiment result from disk.

        :param exp_dir: The directory from where the experiment results are restored.
        :param max_env_step: The maximum number of environment steps to consider. If None, all data is considered.
            Note: if the experiments have different numbers of steps, the minimum number is used.
        """
        test_episode_returns = []
        train_episode_returns = []
        env_step_at_test = None
        """The number of steps of the test run,
        will try extracting it either from the loaded stats or from loaded arrays."""
        env_step_at_train = None
        """The number of steps of the training run,
        will try extracting it from the loaded stats or from loaded arrays."""

        # TODO: env_step_at_test should not be defined in a loop and overwritten at each iteration
        #  just for retrieving them. We might need a cleaner directory structure.
        for entry in os.scandir(exp_dir):
            if entry.name.startswith(".") or not entry.is_dir():
                continue

            try:
                # TODO: fix
                logger_factory = Experiment.from_directory(entry.path).logger_factory
                # only retrieve logger class to prevent creating another tfevent file
                logger_cls = logger_factory.get_logger_class()
            # Usually this means from low-level API
            except FileNotFoundError:
                log.info(
                    f"Could not find persisted experiment in {entry.path}, using default logger.",
                )
                logger_cls = TensorboardLogger

            data = logger_cls.restore_logged_data(entry.path)
            # TODO: align low-level and high-level dir structure. This is a hack!
            if not data:
                dirs = [
                    d for d in os.listdir(entry.path) if os.path.isdir(os.path.join(entry.path, d))
                ]
                if len(dirs) != 1:
                    raise ValueError(
                        f"Could not restore data from {entry.path}, "
                        f"expected either events or exactly one subdirectory, ",
                    )
                data = logger_cls.restore_logged_data(os.path.join(entry.path, dirs[0]))
            if not data:
                raise ValueError(f"Could not restore data from {entry.path}.")

            if DataScope.TEST not in data or not data[DataScope.TEST]:
                continue
            restored_test_data = data[DataScope.TEST]
            restored_train_data = data[DataScope.TRAIN]

            assert isinstance(restored_test_data, dict)
            assert isinstance(restored_train_data, dict)

            for restored_data, scope in zip(
                [restored_test_data, restored_train_data],
                [DataScope.TEST, DataScope.TRAIN],
                strict=True,
            ):
                if not isinstance(restored_data, dict):
                    raise RuntimeError(
                        f"Expected entry with key {scope} data to be a dictionary, "
                        f"but got {restored_data=}.",
                    )
            test_data = LoggedCollectStats.from_data_dict(restored_test_data)
            train_data = LoggedCollectStats.from_data_dict(restored_train_data)

            if test_data.returns_stat is not None:
                test_episode_returns.append(test_data.returns_stat.mean)
                env_step_at_test = test_data.env_step

            if train_data.returns_stat is not None:
                train_episode_returns.append(train_data.returns_stat.mean)
                env_step_at_train = train_data.env_step

        test_data_found = True
        train_data_found = True
        if not test_episode_returns or env_step_at_test is None:
            log.warning(f"No test experiment data found in {exp_dir}.")
            test_data_found = False
        if not train_episode_returns or env_step_at_train is None:
            log.warning(f"No train experiment data found in {exp_dir}.")
            train_data_found = False

        if not test_data_found and not train_data_found:
            raise RuntimeError(f"No test or train data found in {exp_dir}.")

        min_train_len = min([len(arr) for arr in train_episode_returns])
        if max_env_step is not None:
            min_train_len = min(min_train_len, max_env_step)
        min_test_len = min([len(arr) for arr in test_episode_returns])
        if max_env_step is not None:
            min_test_len = min(min_test_len, max_env_step)

        assert env_step_at_test is not None
        assert env_step_at_train is not None

        env_step_at_test = env_step_at_test[:min_test_len]
        env_step_at_train = env_step_at_train[:min_train_len]
        if max_env_step:
            # find the index at which the maximum env step is reached with searchsorted
            min_test_len = int(np.searchsorted(env_step_at_test, max_env_step))
            min_train_len = int(np.searchsorted(env_step_at_train, max_env_step))
            env_step_at_test = env_step_at_test[:min_test_len]
            env_step_at_train = env_step_at_train[:min_train_len]

        test_episode_returns = np.array([arr[:min_test_len] for arr in test_episode_returns])
        train_episode_returns = np.array([arr[:min_train_len] for arr in train_episode_returns])

        return cls(
            test_episode_returns_RE=test_episode_returns,
            env_steps_E=env_step_at_test,
            exp_dir=exp_dir,
            train_episode_returns_RE=train_episode_returns,
            env_steps_train_E=env_step_at_train,
        )

    def _get_rliable_data(
        self,
        algo_name: str | None = None,
        score_thresholds: np.ndarray | None = None,
        scope: DataScope = DataScope.TEST,
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """Return the data in the format expected by the rliable library.

        :param algo_name: The name of the algorithm to be shown in the figure legend. If None, the name of the algorithm
            is set to the experiment dir.
        :param score_thresholds: The score thresholds for the performance profile. If None, the thresholds are inferred
            from the minimum and maximum test episode returns.

        :return: A tuple score_dict, env_steps, and score_thresholds.
        """
        if scope == DataScope.TEST:
            env_steps, returns = self.env_steps_E, self.test_episode_returns_RE
        elif scope == DataScope.TRAIN:
            env_steps, returns = self.env_steps_train_E, self.train_episode_returns_RE
        else:
            raise ValueError(f"Invalid scope {scope}, should be either 'TEST' or 'TRAIN'.")
        if score_thresholds is None:
            score_thresholds = np.linspace(
                np.min(returns),
                np.max(returns),
                101,
            )

        if algo_name is None:
            algo_name = os.path.basename(self.exp_dir)

        score_dict = {algo_name: returns}

        return score_dict, env_steps, score_thresholds

    def eval_results(
        self,
        algo_name: str | None = None,
        score_thresholds: np.ndarray | None = None,
        save_plots: bool = False,
        show_plots: bool = True,
        scope: DataScope = DataScope.TEST,
        ax_iqm: plt.Axes | None = None,
        ax_profile: plt.Axes | None = None,
        algo2color: dict[str, str] | None = None,
    ) -> tuple[plt.Figure, plt.Axes, plt.Figure, plt.Axes]:
        """Evaluate the results of an experiment and create a sample efficiency curve and a performance profile.

        :param algo_name: The name of the algorithm to be shown in the figure legend. If None, the name of the algorithm
            is set to the experiment dir.
        :param score_thresholds: The score thresholds for the performance profile. If None, the thresholds are inferred
            from the minimum and maximum test episode returns.
        :param save_plots: If True, the figures are saved to the experiment directory.
        :param show_plots: If True, the figures are shown.
        :param scope: The scope of the evaluation, either 'TEST' or 'TRAIN'.
        :param ax_iqm: The axis to plot the IQM sample efficiency curve on. If None, a new figure is created.
        :param ax_profile: The axis to plot the performance profile on. If None, a new figure is created.
        :param algo2color: A dictionary mapping algorithm names to colors. Useful for plotting
            the evaluations of multiple algorithms in the same figure, e.g., by first creating an ax_iqm and ax_profile
            with one evaluation and then passing them into the other evaluation. Same as the `colors`
            kwarg in the rliable plotting utils.

        :return: The created figures and axes in the order: fig_iqm, ax_iqm, fig_profile, ax_profile.
        """
        score_dict, env_steps, score_thresholds = self._get_rliable_data(
            algo_name,
            score_thresholds,
            scope,
        )

        iqm = lambda scores: sst.trim_mean(scores, proportiontocut=0.25, axis=0)
        iqm_scores, iqm_cis = rly.get_interval_estimates(score_dict, iqm)

        # Plot IQM sample efficiency curve
        if ax_iqm is None:
            fig_iqm, ax_iqm = plt.subplots(ncols=1, figsize=(7, 5), constrained_layout=True)
        else:
            fig_iqm = ax_iqm.get_figure()  # type: ignore
        plot_utils.plot_sample_efficiency_curve(
            env_steps,
            iqm_scores,
            iqm_cis,
            algorithms=None,
            xlabel="env step",
            ylabel="IQM episode return",
            ax=ax_iqm,
            colors=algo2color,
        )
        if show_plots:
            plt.show(block=False)

        if save_plots:
            iqm_sample_efficiency_curve_path = os.path.abspath(
                os.path.join(
                    self.exp_dir,
                    "iqm_sample_efficiency_curve.png",
                ),
            )
            log.info(f"Saving iqm sample efficiency curve to {iqm_sample_efficiency_curve_path}.")
            fig_iqm.savefig(iqm_sample_efficiency_curve_path)

        final_score_dict = {algo: returns[:, [-1]] for algo, returns in score_dict.items()}
        score_distributions, score_distributions_cis = rly.create_performance_profile(
            final_score_dict,
            score_thresholds,
        )

        # Plot score distributions
        if ax_profile is None:
            fig_profile, ax_profile = plt.subplots(ncols=1, figsize=(7, 5), constrained_layout=True)
        else:
            fig_profile = ax_profile.get_figure()  # type: ignore
        plot_utils.plot_performance_profiles(
            score_distributions,
            score_thresholds,
            performance_profile_cis=score_distributions_cis,
            xlabel=r"Episode return $(\tau)$",
            ax=ax_profile,
        )

        if save_plots:
            profile_curve_path = os.path.abspath(
                os.path.join(self.exp_dir, "performance_profile.png"),
            )
            log.info(f"Saving performance profile curve to {profile_curve_path}.")
            fig_profile.savefig(profile_curve_path)
        if show_plots:
            plt.show(block=False)

        return fig_iqm, ax_iqm, fig_profile, ax_profile


def load_and_eval_experiments(
    log_dir: str,
    show_plots: bool = True,
    save_plots: bool = True,
    scope: DataScope | Literal["both"] = DataScope.TEST,
    max_env_step: int | None = None,
) -> RLiableExperimentResult:
    """Evaluate the experiments in the given log directory using the rliable API and return the loaded results object.

    If neither `show_plots` nor `save_plots` is set to `True`, this is equivalent to just loading the results from disk.

    :param log_dir: The directory containing the experiment results.
    :param show_plots: whether to display plots.
    :param save_plots: whether to save plots to the `log_dir`.
    :param scope: The scope of the evaluation, either 'test', 'train' or 'both'.
    :param max_env_step: The maximum number of environment steps to consider. If None, all data is considered.
            Note: if the experiments have different numbers of steps, the minimum number is used.
    """
    rliable_result = RLiableExperimentResult.load_from_disk(log_dir, max_env_step=max_env_step)
    if scope == "both":
        for scope in [DataScope.TEST, DataScope.TRAIN]:
            rliable_result.eval_results(show_plots=True, save_plots=True, scope=scope)
    else:
        rliable_result.eval_results(show_plots=show_plots, save_plots=save_plots, scope=scope)
    return rliable_result
