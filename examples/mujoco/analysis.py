#!/usr/bin/env python3

import argparse
import re
from collections import defaultdict

import numpy as np
from tabulate import tabulate
from tools import csv2numpy, find_all_files, group_files


def numerical_analysis(root_dir, xlim, norm=False):
    file_pattern = re.compile(r".*/test_reward_\d+seeds.csv$")
    norm_group_pattern = re.compile(r"(/|^)\w+?\-v(\d|$)")
    output_group_pattern = re.compile(r".*?(?=(/|^)\w+?\-v\d)")
    csv_files = find_all_files(root_dir, file_pattern)
    norm_group = group_files(csv_files, norm_group_pattern)
    output_group = group_files(csv_files, output_group_pattern)
    # calculate numerical outcome for each csv_file (y/std integration max_y, final_y)
    results = defaultdict(list)
    for f in csv_files:
        result = csv2numpy(f)
        if norm:
            result = np.stack(
                [
                    result["env_step"],
                    result["reward"] - result["reward"][0],
                    result["reward:shaded"],
                ],
            )
        else:
            result = np.stack([result["env_step"], result["reward"], result["reward:shaded"]])

        if result[0, -1] < xlim:
            continue

        final_rew = np.interp(xlim, result[0], result[1])
        final_rew_std = np.interp(xlim, result[0], result[2])
        result = result[:, result[0] <= xlim]

        if len(result) == 0:
            continue

        if result[0, -1] < xlim:
            last_line = np.array([xlim, final_rew, final_rew_std]).reshape(3, 1)
            result = np.concatenate([result, last_line], axis=-1)

        max_id = np.argmax(result[1])
        results["name"].append(f)
        results["final_reward"].append(result[1, -1])
        results["final_reward_std"].append(result[2, -1])
        results["max_reward"].append(result[1, max_id])
        results["max_std"].append(result[2, max_id])
        results["reward_integration"].append(np.trapz(result[1], x=result[0]))
        results["reward_std_integration"].append(np.trapz(result[2], x=result[0]))

    results = {k: np.array(v) for k, v in results.items()}
    print(tabulate(results, headers="keys"))

    if norm:
        # calculate normalized numerical outcome for each csv_file group
        for _, fs in norm_group.items():
            mask = np.isin(results["name"], fs)
            for k, v in results.items():
                if k == "name":
                    continue
                v[mask] = v[mask] / max(v[mask])
        # Add all numerical results for each outcome group
        group_results = defaultdict(list)
        for g, fs in output_group.items():
            group_results["name"].append(g)
            mask = np.isin(results["name"], fs)
            group_results["num"].append(sum(mask))
            for k in results:
                if k == "name":
                    continue
                group_results[k + ":norm"].append(results[k][mask].mean())
        # print all outputs for each csv_file and each outcome group
        print()
        print(tabulate(group_results, headers="keys"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xlim",
        type=int,
        default=1000000,
        help="x-axis limitation (default: 1000000)",
    )
    parser.add_argument("--root-dir", type=str)
    parser.add_argument(
        "--norm",
        action="store_true",
        help="Normalize all results according to environment.",
    )
    args = parser.parse_args()
    numerical_analysis(args.root_dir, args.xlim, norm=args.norm)
