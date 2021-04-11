import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator
import json
from contextlib import ExitStack
import gym
import re
import csv
import os
import argparse

def find_all_files(root_dir, pattern):
    "find all files in root dir according to relative pattern"
    file_list = []
    for dirname, _, files in os.walk(root_dir):
        for f in files:
            absolute_path = os.path.join(dirname, f)
            if re.match(pattern, absolute_path):
                file_list.append(absolute_path)
    return file_list

def convert_tfevents_to_csv(root_dir = './', pattern=re.compile(".*tfevents[^/]*$")):
    """recursively find all tfevent file under root_dir, and create
    a csv file to track test/rew. This function assumes that 
    there is at most one tfevents file in each directory and will add suffix
    to that directory.#TODO this need to be optimized later
    you can use 
    rl_plotter --save --avg_group --shaded_std --filename=test_rew --smooth=0
    to create standard rl reward graph.
    for more detail, please refer to https://github.com/gxywy/rl-plotter
    """
    print("Converting all tfevents files under {} ...".format(root_dir))
    tfevent_files = []
    for dirname, _, files in os.walk(root_dir):
        for f in files:
            if 'tfevents' in f:
                tfevent_files.append(os.path.join(dirname, f))
                break
    tfevent_files = find_all_files(root_dir, pattern)
    sorted_files = []
    for tfevent_file in tfevent_files:
        print("handling " + tfevent_file)
        ea=event_accumulator.EventAccumulator(tfevent_file) 
        ea.Reload()
        initial_time = ea._first_event_timestamp
        output_file = os.path.join(os.path.split(tfevent_file)[0], "test_rew.csv")
        sorted_files.append(output_file)
        with open(output_file, 'w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=('env_step', 'time', 'rew'))
            writer.writeheader()
            test_rews = ea.scalars.Items('test/rew')
            for test_rew in test_rews:
                episode_info = {'rew': round(test_rew.value, 4), 'env_step': round(test_rew.step, 4), 'time': round(test_rew.wall_time - initial_time, 4)}
                writer.writerow(episode_info)
    return sorted_files

def concatenate_csv(root_dir = './', output_path = None, pattern=re.compile(".*/test_rew.csv$")):
    """for all algorithm-env pair in root_dir, concatenate all test/rew under a single csv file"""
    csv_files = find_all_files(root_dir, pattern)
    assert len(csv_files) >= 1
    # readers = [csv.DictReader(open(csv_file, newline='')) for csv_file in csv_files]
    relative_paths = [os.path.relpath(csv_file, root_dir) for csv_file in csv_files]
    with ExitStack() as stack:
        if output_path is None:
            output_path = os.path.join(root_dir, "test_rew_{}seeds.csv".format(len(csv_files)))
            print("Concatenating {}.".format(output_path))

        readers = [csv.DictReader(stack.enter_context(open(csv_file, newline=''))) for csv_file in csv_files]
        fieldnames = ['env_step', 'rew', 'rew:shaded'] + ['rew:' + p for p in relative_paths]
        output_file = stack.enter_context(open(output_path, 'w'))
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        line_num = 0
        for rows in zip(*readers):
            assert len(set(row['env_step'] for row in rows)) == 1
            rew_list = [round(float(row['rew']), 4) for row in rows]
            episode_info = dict(zip(fieldnames[3:], rew_list))
            episode_info['env_step'] = rows[0]['env_step']
            episode_info['rew'] = round(np.mean(rew_list), 4)
            episode_info['rew:shaded'] = round(np.std(rew_list), 4)
            writer.writerow(episode_info)
            line_num += 1
        # summary
        print("{} is generated, concatenating {} data lines from {} files.".format(output_path, line_num, len(csv_files)))

def group_files(file_list, pattern):
    res = {}
    for f in file_list:
        match = re.search(pattern, f)
        key = match.group() if match else ''
        if key not in res:
            res[key] = []
        res[key].append(f)
    return res

def analysis_data(root_dir, pattern="^.*?(?=/[^/]*seed_\d.*?/)"):
    sorted_files = convert_tfevents_to_csv(root_dir)
    dirs = list(group_files(sorted_files, pattern).keys())
    for dir in dirs:
        concatenate_csv(dir)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default=None)
    parser.add_argument('--concatenate_pattern', type=str, default="^.*?(?=/[^/]*seed_\d.*?/)")
    args = parser.parse_args()
    args.root_dir = "/mfs/huayu/mujoco_benchmark"
    dirs = list(group_files(find_all_files(args.root_dir, r".*/test_rew.csv$"), args.concatenate_pattern).keys())
    for dir in dirs:
        concatenate_csv(dir)
    # analysis_data(args.root_dir, pattern=args.concatenate_pattern)
    