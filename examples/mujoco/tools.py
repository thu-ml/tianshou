#!/usr/bin/env python3

import os
import re
import csv
import tqdm
import argparse
import numpy as np
from typing import Dict, List, Union
from tensorboard.backend.event_processing import event_accumulator
from collections import defaultdict


def find_all_files(root_dir: str, pattern: re.Pattern) -> List[str]:
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
        key = match.group() if match else ''
        res[key].append(f)
    return res


def csv2numpy(csv_file):
    csv_dict = defaultdict(list)
    reader = csv.DictReader(open(csv_file))
    for row in reader:
        for k, v in row.items():
            csv_dict[k].append(eval(v))
    return {k: np.array(v) for k, v in csv_dict.items()}


def convert_tfevents_to_csv(
    root_dir: str, refresh: bool = False
) -> Dict[str, np.ndarray]:
    """Recursively convert test/rew from all tfevent file under root_dir to csv.

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
            output_file = os.path.join(os.path.split(tfevent_file)[0], "test_rew.csv")
            if os.path.exists(output_file) and not refresh:
                content = list(csv.reader(open(output_file, "r")))
                if content[0] == ["env_step", "rew", "time"]:
                    for i in range(1, len(content)):
                        content[i] = list(map(eval, content[i]))
                    result[output_file] = content
                    continue
            ea = event_accumulator.EventAccumulator(tfevent_file)
            ea.Reload()
            initial_time = ea._first_event_timestamp
            content = [["env_step", "rew", "time"]]
            for test_rew in ea.scalars.Items("test/rew"):
                content.append([
                    round(test_rew.step, 4),
                    round(test_rew.value, 4),
                    round(test_rew.wall_time - initial_time, 4),
                ])
            csv.writer(open(output_file, 'w')).writerows(content)
            result[output_file] = content
    return result


def merge_csv(
    csv_files: List[List[Union[str, int, float]]],
    root_dir: str,
    remove_zero: bool = False,
) -> None:
    """Merge result in csv_files into a single csv file."""
    assert len(csv_files) > 0
    if remove_zero:
        for k, v in csv_files.items():
            if v[1][0] == 0:
                v.pop(1)
    sorted_keys = sorted(csv_files.keys())
    sorted_values = [csv_files[k][1:] for k in sorted_keys]
    content = [["env_step", "rew", "rew:shaded"] + list(map(
        lambda f: "rew:" + os.path.relpath(f, root_dir), sorted_keys))]
    for rows in zip(*sorted_values):
        array = np.array(rows)
        assert len(set(array[:, 0])) == 1, (set(array[:, 0]), array[:, 0])
        line = [rows[0][0], round(array[:, 1].mean(), 4), round(array[:, 1].std(), 4)]
        line += array[:, 1].tolist()
        content.append(line)
    output_path = os.path.join(root_dir, f"test_rew_{len(csv_files)}seeds.csv")
    print(f"Output merged csv file to {output_path} with {len(content[1:])} lines.")
    csv.writer(open(output_path, "w")).writerows(content)


def numerical_anysis(root_dir: str, xlim: int, norm: bool = False) -> None:
    file_pattern = r".*/test_rew_\d+seeds.csv$"
    norm_group_pattern = r"(/|^)\w+?\-v(\d|$)"
    output_group_pattern = r".*?(?=(/|^)\w+?\-v\d)"
    csv_files = find_all_files(root_dir, re.compile(file_pattern))
    norm_group = group_files(csv_files, norm_group_pattern)
    output_group = group_files(csv_files, output_group_pattern)
    # calculate numerical outcome for each csv_file (y/std integration max_y, final_y)
    results = {}
    for f in csv_files:
        # reader = csv.DictReader(open(f, newline=''))
        # result = []
        # for row in reader:
        #     result.append([row['env_step'], row['rew'], row['rew:shaded']])
        # result = np.array(result).T
        # iclip = np.searchsorted(result[0], xlim)
        result = csv2numpy(f)
        if norm:
            result = np.stack((
                result['env_step'],
                result['rew'] - result['rew'][0],
                result['rew:shaded']))
        else:
            result = np.stack((
                result['env_step'], result['rew'], result['rew:shaded']))
        iclip = np.searchsorted(result[0], xlim)

        if iclip == 0 or iclip == len(result[0]):
            results[f] = None
            continue
        else:
            results[f] = {}
        result = result[:, :iclip + 1]
        final_rew = np.interp(xlim, result[0], result[1])
        final_rew_std = np.interp(xlim, result[0], result[2])
        result[0, iclip] = xlim
        result[1, iclip] = final_rew
        result[2, iclip] = final_rew_std
        results[f]['final_reward'] = final_rew.astype(float)
        results[f]['final_reward_std'] = final_rew_std.astype(float)
        max_id = np.argmax(result[1])
        max_rew = result[1][max_id]
        max_std = result[2][max_id]

        results[f]['max_reward'] = max_rew.astype(float)
        results[f]['max_std'] = max_std.astype(float)
        rew_integration = np.trapz(result[1], x=result[0])
        results[f]['reward_integration'] = rew_integration.astype(float)
        std_integration = np.trapz(result[2], x=result[0])
        results[f]['reward_std_integration'] = std_integration.astype(float)

    for f, numerical_result in results.items():
        print("*******  " + f + ":")
        print(numerical_result)

    if norm:
        # calculate normalised numerical outcome for each csv_file group
        for _, fs in norm_group.items():
            maxres = defaultdict(lambda: -np.inf)
            # find max for each key
            for f in fs:
                if not results[f]:
                    continue
                for k, v in results[f].items():
                    maxres[k] = v if maxres[k] < v else maxres[k]
            # add normalised numerical outcome
            for f in fs:
                if not results[f]:
                    continue
                new_dict = results[f].copy()
                for k, v in results[f].items():
                    new_dict[k + ":normalized"] = v / maxres[k]
                results[f] = new_dict
        # Add all numerical results for each outcome group
        output_group
        group_results = {}
        for g, fs in output_group.items():
            group_results[g] = defaultdict(lambda: 0)
            group_n = 0
            for f in fs:
                if not results[f]:
                    continue
                group_n += 1
                for k, v in results[f].items():
                    group_results[g][k] += v
            for k, v in group_results[g].items():
                group_results[g][k] = v / group_n
            group_results[g]['group_n'] += group_n
        # print all outputs for each csv_file and each outcome group

        for g, numerical_result in group_results.items():
            print("*******  " + g + ":")
            print(numerical_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sp = parser.add_subparsers(dest='action')

    merge_parser = sp.add_parser('merge')
    merge_parser.add_argument(
        '--refresh', action="store_true",
        help="Re-generate all csv files instead of using existing one.")
    merge_parser.add_argument(
        '--remove-zero', action="store_true",
        help="Remove the data point of env_step == 0.")
    merge_parser.add_argument('--root-dir', type=str)

    analysis_parser = sp.add_parser('analysis')
    analysis_parser.add_argument('--xlim', type=int, default=1000000,
                                 help='x-axis limitation (default: 1000000)')
    analysis_parser.add_argument('--root-dir', type=str)
    analysis_parser.add_argument(
        '--norm', action="store_true",
        help="Normalize all results according to environment.")
    args = parser.parse_args()

    if args.action == "merge":
        csv_files = convert_tfevents_to_csv(args.root_dir, args.refresh)
        merge_csv(csv_files, args.root_dir, args.remove_zero)
    elif args.action == "analysis":
        numerical_anysis(args.root_dir, args.xlim, norm=args.norm)
