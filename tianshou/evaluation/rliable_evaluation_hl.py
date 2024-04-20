"""The rliable-evaluation module provides a high-level interface to evaluate the results of an experiment with multiple runs
on different seeds using the rliable library. The API is experimental and subject to change!.
"""

import os
from dataclasses import asdict, dataclass, fields

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sst
from rliable import library as rly
from rliable import plot_utils

from tianshou.highlevel.experiment import Experiment
from tianshou.utils import logging
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

    env_steps_E: np.ndarray
    """The number of environment steps at which the test episodes were evaluated."""

    @classmethod
    def load_from_disk(cls, exp_dir: str) -> "RLiableExperimentResult":
        """Load the experiment result from disk.

        :param exp_dir: The directory from where the experiment results are restored.
        """
        test_episode_returns = []
        env_step_at_test = None

        # TODO: env_step_at_test should not be defined in a loop and overwritten at each iteration
        #  just for retrieving them. We might need a cleaner directory structure.
        for entry in os.scandir(exp_dir):
            if entry.name.startswith(".") or not entry.is_dir():
                continue

            exp = Experiment.from_directory(entry.path)
            logger = exp.logger_factory.create_logger(
                entry.path,
                entry.name,
                None,
                asdict(exp.config),
            )
            data = logger.restore_logged_data(entry.path)

            if DataScope.TEST.value not in data or not data[DataScope.TEST.value]:
                continue
            restored_test_data = data[DataScope.TEST.value]
            if not isinstance(restored_test_data, dict):
                raise RuntimeError(
                    f"Expected entry with key {DataScope.TEST.value} data to be a dictionary, "
                    f"but got {restored_test_data=}.",
                )
            test_data = LoggedCollectStats.from_data_dict(restored_test_data)

            if test_data.returns_stat is None:
                continue
            test_episode_returns.append(test_data.returns_stat.mean)
            env_step_at_test = test_data.env_step

        if not test_episode_returns or env_step_at_test is None:
            raise ValueError(f"No experiment data found in {exp_dir}.")

        return cls(
            test_episode_returns_RE=np.array(test_episode_returns),
            env_steps_E=np.array(env_step_at_test),
            exp_dir=exp_dir,
        )

    def _get_rliable_data(
        self,
        algo_name: str | None = None,
        score_thresholds: np.ndarray | None = None,
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """Return the data in the format expected by the rliable library.

        :param algo_name: The name of the algorithm to be shown in the figure legend. If None, the name of the algorithm
            is set to the experiment dir.
        :param score_thresholds: The score thresholds for the performance profile. If None, the thresholds are inferred
            from the minimum and maximum test episode returns.

        :return: A tuple score_dict, env_steps, and score_thresholds.
        """
        if score_thresholds is None:
            score_thresholds = np.linspace(
                np.min(self.test_episode_returns_RE),
                np.max(self.test_episode_returns_RE),
                101,
            )

        if algo_name is None:
            algo_name = os.path.basename(self.exp_dir)

        score_dict = {algo_name: self.test_episode_returns_RE}

        return score_dict, self.env_steps_E, score_thresholds

    def eval_results(
        self,
        algo_name: str | None = None,
        score_thresholds: np.ndarray | None = None,
        save_plots: bool = False,
        show_plots: bool = True,
    ) -> tuple[plt.Figure, plt.Axes, plt.Figure, plt.Axes]:
        """Evaluate the results of an experiment and create a sample efficiency curve and a performance profile.

        :param algo_name: The name of the algorithm to be shown in the figure legend. If None, the name of the algorithm
            is set to the experiment dir.
        :param score_thresholds: The score thresholds for the performance profile. If None, the thresholds are inferred
            from the minimum and maximum test episode returns.
        :param save_plots: If True, the figures are saved to the experiment directory.
        :param show_plots: If True, the figures are shown.

        :return: The created figures and axes.
        """
        score_dict, env_steps, score_thresholds = self._get_rliable_data(
            algo_name,
            score_thresholds,
        )

        iqm = lambda scores: sst.trim_mean(scores, proportiontocut=0.25, axis=0)
        iqm_scores, iqm_cis = rly.get_interval_estimates(score_dict, iqm)

        # Plot IQM sample efficiency curve
        fig_iqm, ax_iqm = plt.subplots(ncols=1, figsize=(7, 5), constrained_layout=True)
        plot_utils.plot_sample_efficiency_curve(
            env_steps,
            iqm_scores,
            iqm_cis,
            algorithms=None,
            xlabel="env step",
            ylabel="IQM episode return",
            ax=ax_iqm,
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
        fig_profile, ax_profile = plt.subplots(ncols=1, figsize=(7, 5), constrained_layout=True)
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
