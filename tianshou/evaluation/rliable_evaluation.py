"""The rliable-evaluation module provides a high-level interface to evaluate the results of an experiment with multiple runs
on different seeds using the rliable library. The API is experimental and subject to change!.
"""

import json
import os
from collections.abc import Iterator
from dataclasses import asdict, dataclass, fields
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sst
from rliable import library as rly
from rliable import plot_utils
from sensai.util import logging

from tianshou.utils import TensorboardLogger
from tianshou.utils.logger.logger_base import DataScope

log = logging.getLogger(__name__)


@dataclass
class EvaluationSequenceEntry:
    """A single entry in an evaluation sequence, representing data collected at a fixed environment
    step.
    """

    # the structure expected in benchmark.js
    env_step: int
    """The number of environment steps at which the evaluation was performed."""
    rew: float
    """The mean episode return at the given env_step. Called `rew` (confusingly) to be consistent with Tianshou's
    internal naming conventions."""
    rew_std: float
    """The standard deviation of the episode returns at the given env_step, computed from multiple runs."""
    iqm: float
    """The interquartile mean (IQM) of the episode returns at the given env_step, computed from multiple runs."""
    iqm_confidence_interval: tuple[float, float]
    """The 95% confidence interval of the IQM of the episode returns at the given env_step."""


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
        dataclass_data = {}
        field_names = [f.name for f in fields(cls)]
        for k, v in data.items():
            if k not in field_names:
                log.info(
                    f"Key {k} in data dict is not a valid field of LoggedCollectStats, ignoring it.",
                )
                continue
            if isinstance(v, dict):
                v = LoggedSummaryData(**v)
            dataclass_data[k] = v
        return cls(**dataclass_data)


@dataclass
class MultiRunExperimentResult:
    """The result of multiple runs of an experiment (runs usually just differing by random seeds)
    that can be used with the rliable library.

    Glossary:
     - R: number of runs (typically, equal to the number of different seeds)
     - E: number of environment steps at which evaluation results were computed, i.e., the evaluation points
          n_1, n_2, ..., n_E
    """

    exp_dir: str
    """The base directory where each sub-directory contains the results of one experiment run."""

    exp_name: str
    """The name of the experiment, typically the name of the algorithm or the experiment directory basename."""

    test_episode_returns_RE: np.ndarray
    """The test episode returns for each run of the experiment, where each row corresponds to one run."""

    training_episode_returns_RE: np.ndarray
    """The training episode returns for each run of the experiment, where each row corresponds to one run."""

    test_env_steps_E: np.ndarray
    """The environment steps at which the test episodes were evaluated."""

    training_env_steps_E: np.ndarray
    """The environment steps at which the training episodes were evaluated."""

    @classmethod
    def load_from_disk(
        cls,
        exp_dir: str,
        exp_name: str | None = None,
        max_env_step: int | None = None,
    ) -> "MultiRunExperimentResult":
        """Load the experiment result from disk.

        :param exp_dir: The directory from where the experiment results are restored.
        :param exp_name: The name of the experiment. If not passed, will be inferred from the experiment directory name.
        :param max_env_step: The maximum number of environment steps to consider. If None, all data is considered.
            Note: if the experiments have different numbers of steps, the minimum number is used.
        """
        test_episode_returns_RE = []
        training_episode_returns_RE = []
        test_env_steps_E = None
        """The number of steps of the test run,
        will try extracting it either from the loaded stats or from loaded arrays."""
        training_env_steps_E = None
        """The number of steps of the training run,
        will try extracting it from the loaded stats or from loaded arrays."""

        if exp_name is None:
            exp_name = os.path.basename(os.path.normpath(exp_dir))

        from tianshou.highlevel.experiment import Experiment
        # TODO: test_env_steps_E should not be defined in a loop and overwritten at each iteration
        #  just for retrieving them. We might need a cleaner directory structure.
        for entry in os.scandir(exp_dir):
            if entry.name.startswith(".") or not entry.is_dir():
                continue

            try:
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
            if not data:
                raise ValueError(f"Could not restore data from {entry.path}.")

            if DataScope.TEST not in data or not data[DataScope.TEST]:
                continue
            restored_test_data = data[DataScope.TEST]
            restored_train_data = data[DataScope.TRAINING]

            assert isinstance(restored_test_data, dict)
            assert isinstance(restored_train_data, dict)

            for restored_data, scope in zip(
                [restored_test_data, restored_train_data],
                [DataScope.TEST, DataScope.TRAINING],
                strict=True,
            ):
                if not isinstance(restored_data, dict):
                    raise RuntimeError(
                        f"Expected entry with key {scope} data to be a dictionary, "
                        f"but got {restored_data=}.",
                    )
            test_data = LoggedCollectStats.from_data_dict(restored_test_data)
            training_data = LoggedCollectStats.from_data_dict(restored_train_data)

            if test_data.returns_stat is not None:
                test_episode_returns_RE.append(test_data.returns_stat.mean)
                test_env_steps_E = test_data.env_step

            if training_data.returns_stat is not None:
                training_episode_returns_RE.append(training_data.returns_stat.mean)
                training_env_steps_E = training_data.env_step

        test_data_found = True
        training_data_found = True
        if not test_episode_returns_RE or test_env_steps_E is None:
            log.warning(f"No test experiment data found in {exp_dir}.")
            test_data_found = False
        if not training_episode_returns_RE or training_env_steps_E is None:
            log.warning(f"No train experiment data found in {exp_dir}.")
            training_data_found = False

        if not test_data_found and not training_data_found:
            raise RuntimeError(f"No test or train data found in {exp_dir}.")

        min_training_data_len = min([len(arr) for arr in training_episode_returns_RE])
        if max_env_step is not None:
            min_training_data_len = min(min_training_data_len, max_env_step)
        min_test_data_len = min([len(arr) for arr in test_episode_returns_RE])
        if max_env_step is not None:
            min_test_data_len = min(min_test_data_len, max_env_step)

        assert test_env_steps_E is not None
        assert training_env_steps_E is not None

        test_env_steps_E = test_env_steps_E[:min_test_data_len]
        training_env_steps_E = training_env_steps_E[:min_training_data_len]
        if max_env_step:
            # find the index at which the maximum env step is reached with searchsorted
            min_test_data_len = int(np.searchsorted(test_env_steps_E, max_env_step))
            min_training_data_len = int(np.searchsorted(training_env_steps_E, max_env_step))
            test_env_steps_E = test_env_steps_E[:min_test_data_len]
            training_env_steps_E = training_env_steps_E[:min_training_data_len]

        test_episode_returns_RE = np.array([arr[:min_test_data_len] for arr in test_episode_returns_RE])
        training_episode_returns_RE = np.array(
            [arr[:min_training_data_len] for arr in training_episode_returns_RE]
        )

        return cls(
            test_episode_returns_RE=test_episode_returns_RE,
            test_env_steps_E=test_env_steps_E,
            exp_dir=exp_dir,
            exp_name=exp_name,
            training_episode_returns_RE=training_episode_returns_RE,
            training_env_steps_E=training_env_steps_E,
        )

    def _get_env_steps_and_returns(
        self,
        scope: DataScope = DataScope.TEST,
    ) -> tuple[np.ndarray, np.ndarray]:
        if scope == DataScope.TEST:
            return self.test_env_steps_E, self.test_episode_returns_RE
        elif scope == DataScope.TRAINING:
            return self.training_env_steps_E, self.training_episode_returns_RE
        else:
            raise ValueError(f"Invalid scope {scope}, should be either 'TEST' or 'TRAINING'.")

    def _get_data_in_rliable_format(
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

        :return: A tuple score_dict (algo_name->returns), env_steps, and score_thresholds.
        """
        env_steps, returns = self._get_env_steps_and_returns(scope=scope)
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

    def _compute_iqm_scores(
        self,
        scope: DataScope = DataScope.TEST,
    ) -> tuple[dict, dict]:
        """Compute the IQM scores and confidence intervals for the experiment in a format
        expected by the rliable library.

        :param scope: The scope of the evaluation, either 'TEST' or 'TRAIN'.

        :return: A tuple of dicts, each with a single entry,
            (self.exp_name->iqm_scores, self.exp_name->iqm_confidence_intervals), where
            confidence intervals is an array of shape (2 x E)
            where the first row contains the lower bounds while the second row contains
            the upper bound of 95% CIs.
        """
        score_dict, _, _ = self._get_data_in_rliable_format(
            algo_name=self.exp_name,
            score_thresholds=None,
            scope=scope,
        )

        compute_iqm = lambda scores: sst.trim_mean(scores, proportiontocut=0.25, axis=0)
        return rly.get_interval_estimates(score_dict, compute_iqm)

    def eval_results(
        self,
        algo_name: str | None = None,
        score_thresholds: np.ndarray | None = None,
        save_as_json: bool = True,
        save_plots: bool = True,
        show_plots: bool = True,
        scope: DataScope = DataScope.TEST,
        ax_iqm_sample_efficiency: plt.Axes | None = None,
        ax_performance_profile: plt.Axes | None = None,
        algo2color: dict[str, str] | None = None,
    ) -> tuple[plt.Figure, plt.Axes, plt.Figure, plt.Axes]:
        """Evaluate the results of an experiment and create a sample efficiency curve and a performance profile.

        :param algo_name: The name of the algorithm to be shown in the figure legend. If None, the name of the algorithm
            is set to the experiment dir.
        :param score_thresholds: The score thresholds for the performance profile. If None, they will be inferred
            from the minimum and maximum test episode returns.
        :param save_as_json: whether to save the evaluation results as a JSON file (in a format compatible by the Tianshou
          benchmarking visualization) in the experiment directory.
        :param save_plots: whether to save the plots to the experiment directory.
        :param show_plots: whether to display the plots.
        :param scope: The scope of the evaluation, either 'TEST' or 'TRAIN'.
        :param ax_iqm_sample_efficiency: The axis to plot the IQM sample efficiency curve on. If None, a new figure is created.
        :param ax_performance_profile: The axis to plot the performance profile on. If None, a new figure is created.
        :param algo2color: A dictionary mapping algorithm names to colors. Useful for plotting
            the evaluations of multiple algorithms in the same figure, e.g., by first creating an ax_iqm and ax_profile
            with one evaluation and then passing them into the other evaluation. Same as the `colors`
            kwarg in the rliable plotting utils.

        :return: The created figures and axes in the order: fig_iqm, ax_iqm, fig_profile, ax_profile.
        """
        iqm_scores, iqm_confidence_intervals = self._compute_iqm_scores(scope=scope)
        # Plot IQM sample efficiency curve
        if ax_iqm_sample_efficiency is None:
            fig_iqm_sample_efficiency, ax_iqm_sample_efficiency = plt.subplots(
                ncols=1, figsize=(7, 5), constrained_layout=True
            )
        else:
            fig_iqm_sample_efficiency = ax_iqm_sample_efficiency.get_figure()  # type: ignore
        score_dict, env_steps, score_thresholds = self._get_data_in_rliable_format(
            algo_name=algo_name,
            score_thresholds=score_thresholds,
            scope=scope,
        )
        plot_utils.plot_sample_efficiency_curve(
            env_steps,
            iqm_scores,
            iqm_confidence_intervals,
            algorithms=None,
            xlabel="env step",
            ylabel="IQM episode return",
            ax=ax_iqm_sample_efficiency,
            colors=algo2color,
        )
        if show_plots:
            plt.show(block=False)

        if save_plots:
            iqm_sample_efficiency_curve_path = os.path.abspath(
                os.path.join(
                    self.exp_dir,
                    f"iqm_sample_efficiency_curve_{scope.value}.png",
                ),
            )
            log.info(f"Saving iqm sample efficiency curve to {iqm_sample_efficiency_curve_path}.")
            fig_iqm_sample_efficiency.savefig(iqm_sample_efficiency_curve_path)

        final_score_dict = {algo: returns[:, [-1]] for algo, returns in score_dict.items()}
        score_distributions, score_distributions_cis = rly.create_performance_profile(
            final_score_dict,
            score_thresholds,
        )

        # Plot score distributions
        if ax_performance_profile is None:
            fig_performance_profile, ax_performance_profile = plt.subplots(
                ncols=1, figsize=(7, 5), constrained_layout=True
            )
        else:
            fig_performance_profile = ax_performance_profile.get_figure()  # type: ignore
        plot_utils.plot_performance_profiles(
            score_distributions,
            score_thresholds,
            performance_profile_cis=score_distributions_cis,
            xlabel=r"Episode return $(\tau)$",
            ax=ax_performance_profile,
        )

        if save_plots:
            profile_curve_path = os.path.abspath(
                os.path.join(self.exp_dir, f"performance_profile_{scope.value}.png"),
            )
            log.info(f"Saving performance profile curve to {profile_curve_path}.")
            fig_performance_profile.savefig(profile_curve_path)
        if show_plots:
            plt.show(block=False)

        if save_as_json:
            json_path = os.path.abspath(
                os.path.join(self.exp_dir, f"rliable_evaluation_{scope.value.lower()}.json"),
            )
            log.info(f"Saving rliable evaluation results to {json_path}.")
            eval_results_dict_sequence = [
                asdict(eval_entry) for eval_entry in self.to_evaluation_sequence(scope=scope)
            ]
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(eval_results_dict_sequence, f, indent=4)

        return (
            fig_iqm_sample_efficiency,
            ax_iqm_sample_efficiency,
            fig_performance_profile,
            ax_performance_profile,
        )

    def to_evaluation_sequence(
        self, scope: DataScope = DataScope.TEST
    ) -> Iterator[EvaluationSequenceEntry]:
        """Convert the experiment result to EvaluationSequence.

        :param scope: The scope of the evaluation, either 'TEST' or 'TRAIN'.

        :return: The rliable EvaluationSequence.
        """
        env_steps_E, returns_RE = self._get_env_steps_and_returns(scope=scope)
        iqm_scores_dict, iqm_confidence_intervals_dict = self._compute_iqm_scores(scope=scope)
        iqm_scores = iqm_scores_dict[self.exp_name]
        iqm_confidence_intervals = iqm_confidence_intervals_dict[self.exp_name]
        for i, env_step in enumerate(env_steps_E):
            returns_R = returns_RE[:, i]
            returns_mean, returns_std = np.mean(returns_R), np.std(returns_R)
            yield EvaluationSequenceEntry(
                env_step=env_step,
                rew_std=returns_std,
                rew=returns_mean,
                iqm=iqm_scores[i],
                iqm_confidence_interval=tuple(iqm_confidence_intervals[:, i]),
            )


def load_and_eval_experiment(
    log_dir: str,
    show_plots: bool = True,
    save_plots: bool = True,
    save_as_json: bool = True,
    scope: DataScope | Literal["both"] = DataScope.TEST,
    max_env_step: int | None = None,
) -> MultiRunExperimentResult:
    """Evaluate the experiments in the given log directory using the rliable API and return the loaded results object.
    By default, will persist the evaluation results as plots and JSON files in the experiment directory.

    :param log_dir: The directory containing the experiment results.
    :param show_plots: whether to display plots.
    :param save_plots: whether to save plots to the `log_dir`.
    :param save_as_json: whether to save the evaluation results as a JSON file (in a format compatible by the Tianshou
          benchmarking visualization) in the experiment directory.
    :param scope: The scope of the evaluation (training or test) or 'both'.
    :param max_env_step: The maximum number of environment steps to consider. If None, all data is considered.
            Note: if the experiments have different numbers of steps, the minimum number is used.
    """
    rliable_result = MultiRunExperimentResult.load_from_disk(log_dir, max_env_step=max_env_step)
    scopes = [scope]
    if scope == "both":
        scopes = [DataScope.TEST, DataScope.TRAINING]
    for scope in scopes:
        rliable_result.eval_results(
            show_plots=show_plots,
            save_plots=save_plots,
            save_as_json=save_as_json,
            scope=scope,
        )
    return rliable_result
