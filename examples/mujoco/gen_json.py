#!/usr/bin/env python3

import os
import csv
import sys
import json


def merge(rootdir):
    """format: $rootdir/$algo/*.csv"""
    result = []
    for path, dirnames, filenames in os.walk(rootdir):
        filenames = [f for f in filenames if f.endswith('.csv')]
        if len(filenames) == 0:
            continue
        elif len(filenames) != 1:
            print(f'More than 1 csv found in {path}!')
            continue
        algo = os.path.relpath(path, rootdir).upper()
        reader = csv.DictReader(open(os.path.join(path, filenames[0])))
        for row in reader:
            result.append({
                'env_step': int(row['env_step']),
                'rew': float(row['rew']),
                'rew_std': float(row['rew:shaded']),
                'Agent': algo,
            })
    open(os.path.join(rootdir, 'result.json'), 'w').write(json.dumps(result))


if __name__ == "__main__":
    merge(sys.argv[-1])
