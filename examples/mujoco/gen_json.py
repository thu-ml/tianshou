#!/usr/bin/env python3

import csv
import json
import os
import sys


def merge(rootdir):
    """format: $rootdir/$algo/*.csv."""
    result = []
    for path, _, filenames in os.walk(rootdir):
        filtered_filenames = [f for f in filenames if f.endswith(".csv")]
        if len(filtered_filenames) == 0:
            continue
        if len(filtered_filenames) != 1:
            print(f"More than 1 csv found in {path}!")
            continue
        algo = os.path.relpath(path, rootdir).upper()
        with open(os.path.join(path, filtered_filenames[0])) as f:
            reader = csv.DictReader(f)
            for row in reader:
                result.append(
                    {
                        "env_step": int(row["env_step"]),
                        "rew": float(row["reward"]),
                        "rew_std": float(row["reward:shaded"]),
                        "Agent": algo,
                    },
                )
    with open(os.path.join(rootdir, "result.json"), "w") as f:
        f.write(json.dumps(result))


if __name__ == "__main__":
    merge(sys.argv[-1])
