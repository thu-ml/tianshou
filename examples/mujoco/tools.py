#!/usr/bin/env python3

import argparse
import csv
import os
import re
from collections import defaultdict
from dataclasses import dataclass, asdict

import numpy as np
import tqdm
from tensorboard.backend.event_processing import event_accumulator

from tianshou.highlevel.experiment import Experiment


@dataclass
class RLiableExperimentResult:
    exp_dir: str
    algorithms: list[str]
    score_dict: dict[str, np.ndarray]  # (n_runs x n_epochs + 1)
    env_steps: np.ndarray  # (n_epochs + 1)
    score_thresholds: np.ndarray

    @staticmethod
    def load_from_disk(exp_dir: str, algo_name: str, score_thresholds: np.ndarray | None):
        """Load the experiment result from disk.

        :param exp_dir: The directory from where the experiment results are restored.
        :param algo_name: The name of the algorithm used in the figure legend.
        :param score_thresholds: The thresholds used to create the performance profile.
            If None, it will be created from the test episode returns.
        """
        test_episode_returns = []

        for entry in os.scandir(exp_dir):
            if entry.name.startswith('.'):
                continue

            exp = Experiment.from_directory(entry.path)
            logger = exp.logger_factory.create_logger(entry.path, entry.name, None, asdict(exp.config))
            data = logger.restore_logged_data(entry.path)

            test_data = data['test']

            test_episode_returns.append(test_data['returns_stat']['mean'])
        env_step = test_data['env_step']

        if score_thresholds is None:
            score_thresholds = np.linspace(0.0, np.max(test_episode_returns), 101)

        return RLiableExperimentResult(algorithms=[algo_name],
                                       score_dict={algo_name: np.array(test_episode_returns)},
                                       env_steps=np.array(env_step),
                                       score_thresholds=score_thresholds,
                                       exp_dir=exp_dir)


def eval_results(results: RLiableExperimentResult):
    import matplotlib.pyplot as plt
    import scipy.stats as sst
    import seaborn as sns
    from rliable import library as rly
    from rliable import plot_utils

    iqm = lambda scores: sst.trim_mean(scores, proportiontocut=0.25, axis=0)
    iqm_scores, iqm_cis = rly.get_interval_estimates(
        results.score_dict, iqm, reps=50000)

    # Plot IQM sample efficiency curve
    fig, ax = plt.subplots(ncols=1, figsize=(7, 5))
    plot_utils.plot_sample_efficiency_curve(
        results.env_steps, iqm_scores, iqm_cis, algorithms=results.algorithms,
        xlabel=r'Number of env steps',
        ylabel='IQM episode return',
        ax=ax)
    plt.savefig(os.path.join(results.exp_dir, 'iqm_sample_efficiency_curve.png'))

    final_score_dict = {algo: returns[:, [-1]] for algo, returns in results.score_dict.items()}
    score_distributions, score_distributions_cis = rly.create_performance_profile(
        final_score_dict, results.score_thresholds)

    # Plot score distributions
    fig, ax = plt.subplots(ncols=1, figsize=(7, 5))
    plot_utils.plot_performance_profiles(
        score_distributions, results.score_thresholds,
        performance_profile_cis=score_distributions_cis,
        colors=dict(zip(results.algorithms, sns.color_palette('colorblind'))),
        xlabel=r'Episode return $(\tau)$',
        ax=ax)
    plt.savefig(os.path.join(results.exp_dir, 'performance_profile.png'))


def find_all_files(root_dir, pattern):
    """Find all files under root_dir according to relative pattern."""
    file_list = []
    for dirname, _, files in os.walk(root_dir):
        for f in files:
            absolute_path = os.path.join(dirname, f)
            if re.match(pattern, absolute_path):
                file_list.append(absolute_path)
    return file_list


def group_files(file_list, pattern):
    res = defaultdict(list)
    for f in file_list:
        match = re.search(pattern, f)
        key = match.group() if match else ""
        res[key].append(f)
    return res


def csv2numpy(csv_file):
    csv_dict = defaultdict(list)
    with open(csv_file) as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                csv_dict[k].append(eval(v))
    return {k: np.array(v) for k, v in csv_dict.items()}


def convert_tfevents_to_csv(root_dir, refresh=False):
    """Recursively convert test/reward from all tfevent file under root_dir to csv.

    This function assumes that there is at most one tfevents file in each directory
    and will add suffix to that directory.

    :param bool refresh: re-create csv file under any condition.
    """
    tfevent_files = find_all_files(root_dir, re.compile(r"^.*tfevents.*$"))
    print(f"Converting {len(tfevent_files)} tfevents files under {root_dir} ...")
    result = {}
    with tqdm.tqdm(tfevent_files) as t:
        for tfevent_file in t:
            t.set_postfix(file=tfevent_file)
            output_file = os.path.join(os.path.split(tfevent_file)[0], "test_reward.csv")
            if os.path.exists(output_file) and not refresh:
                with open(output_file) as f:
                    content = list(csv.reader(f))
                if content[0] == ["env_step", "reward", "time"]:
                    for i in range(1, len(content)):
                        content[i] = list(map(eval, content[i]))
                    result[output_file] = content
                    continue
            ea = event_accumulator.EventAccumulator(tfevent_file)
            ea.Reload()
            initial_time = ea._first_event_timestamp
            content = [["env_step", "reward", "time"]]
            for test_reward in ea.scalars.Items("test/reward"):
                content.append(
                    [
                        round(test_reward.step, 4),
                        round(test_reward.value, 4),
                        round(test_reward.wall_time - initial_time, 4),
                    ],
                )
            with open(output_file, "w") as f:
                csv.writer(f).writerows(content)
            result[output_file] = content
    return result


def merge_csv(csv_files, root_dir, remove_zero=False):
    """Merge result in csv_files into a single csv file."""
    assert len(csv_files) > 0
    if remove_zero:
        for v in csv_files.values():
            if v[1][0] == 0:
                v.pop(1)
    sorted_keys = sorted(csv_files.keys())
    sorted_values = [csv_files[k][1:] for k in sorted_keys]
    content = [
        [
            "env_step",
            "reward",
            "reward:shaded",
            *["reward:" + os.path.relpath(f, root_dir) for f in sorted_keys],
        ],
    ]
    for rows in zip(*sorted_values, strict=True):
        array = np.array(rows)
        assert len(set(array[:, 0])) == 1, (set(array[:, 0]), array[:, 0])
        line = [rows[0][0], round(array[:, 1].mean(), 4), round(array[:, 1].std(), 4)]
        line += array[:, 1].tolist()
        content.append(line)
    output_path = os.path.join(root_dir, f"test_reward_{len(csv_files)}seeds.csv")
    print(f"Output merged csv file to {output_path} with {len(content[1:])} lines.")
    with open(output_path, "w") as f:
        csv.writer(f).writerows(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-generate all csv files instead of using existing one.",
    )
    parser.add_argument(
        "--remove-zero",
        action="store_true",
        help="Remove the data point of env_step == 0.",
    )
    parser.add_argument("--root-dir", type=str)
    args = parser.parse_args()

    csv_files = convert_tfevents_to_csv(args.root_dir, args.refresh)
    merge_csv(csv_files, args.root_dir, args.remove_zero)
