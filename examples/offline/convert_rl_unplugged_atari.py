#!/usr/bin/env python3
#
# Adapted from
# https://github.com/deepmind/deepmind-research/blob/master/rl_unplugged/atari.py
#
"""Convert Atari RL Unplugged datasets to HDF5 format.

Examples in the dataset represent SARSA transitions stored during a
DQN training run as described in https://arxiv.org/pdf/1907.04543.

For every training run we have recorded all 50 million transitions corresponding
to 200 million environment steps (4x factor because of frame skipping). There
are 5 separate datasets for each of the 45 games.

Every transition in the dataset is a tuple containing the following features:

* o_t: Observation at time t. Observations have been processed using the
    canonical Atari frame processing, including 4x frame stacking. The shape
    of a single observation is [84, 84, 4].
* a_t: Action taken at time t.
* r_t: Reward after a_t.
* d_t: Discount after a_t.
* o_tp1: Observation at time t+1.
* a_tp1: Action at time t+1.
* extras:
  * episode_id: Episode identifier.
  * episode_return: Total episode return computed using per-step [-1, 1]
      clipping.
"""
import os
from argparse import ArgumentParser

import h5py
import numpy as np
import requests
import tensorflow as tf
from tqdm import tqdm

from tianshou.data import Batch

tf.config.set_visible_devices([], 'GPU')

# 9 tuning games.
TUNING_SUITE = [
    "BeamRider",
    "DemonAttack",
    "DoubleDunk",
    "IceHockey",
    "MsPacman",
    "Pooyan",
    "RoadRunner",
    "Robotank",
    "Zaxxon",
]

# 36 testing games.
TESTING_SUITE = [
    "Alien",
    "Amidar",
    "Assault",
    "Asterix",
    "Atlantis",
    "BankHeist",
    "BattleZone",
    "Boxing",
    "Breakout",
    "Carnival",
    "Centipede",
    "ChopperCommand",
    "CrazyClimber",
    "Enduro",
    "FishingDerby",
    "Freeway",
    "Frostbite",
    "Gopher",
    "Gravitar",
    "Hero",
    "Jamesbond",
    "Kangaroo",
    "Krull",
    "KungFuMaster",
    "NameThisGame",
    "Phoenix",
    "Pong",
    "Qbert",
    "Riverraid",
    "Seaquest",
    "SpaceInvaders",
    "StarGunner",
    "TimePilot",
    "UpNDown",
    "VideoPinball",
    "WizardOfWor",
    "YarsRevenge",
]

# Total of 45 games.
ALL_GAMES = TUNING_SUITE + TESTING_SUITE
URL_PREFIX = "http://storage.googleapis.com/rl_unplugged/atari"


def _filename(run_id: int, shard_id: int, total_num_shards: int = 100) -> str:
    return f"run_{run_id}-{shard_id:05d}-of-{total_num_shards:05d}"


def _decode_frames(pngs: tf.Tensor) -> tf.Tensor:
    """Decode PNGs.

    Args:
      pngs: String Tensor of size (4,) containing PNG encoded images.

    Returns:
      4 84x84 grayscale images packed in a (4, 84, 84) uint8 Tensor.
    """
    # Statically unroll png decoding
    frames = [tf.image.decode_png(pngs[i], channels=1) for i in range(4)]
    # NOTE: to match tianshou's convention for framestacking
    frames = tf.squeeze(tf.stack(frames, axis=0))
    frames.set_shape((4, 84, 84))
    return frames


def _make_tianshou_batch(
    o_t: tf.Tensor,
    a_t: tf.Tensor,
    r_t: tf.Tensor,
    d_t: tf.Tensor,
    o_tp1: tf.Tensor,
    a_tp1: tf.Tensor,
) -> Batch:
    """Create Tianshou batch with offline data.

    Args:
      o_t: Observation at time t.
      a_t: Action at time t.
      r_t: Reward at time t.
      d_t: Discount at time t.
      o_tp1: Observation at time t+1.
      a_tp1: Action at time t+1.

    Returns:
      A tianshou.data.Batch object.
    """
    return Batch(
        obs=o_t.numpy(),
        act=a_t.numpy(),
        rew=r_t.numpy(),
        done=1 - d_t.numpy(),
        obs_next=o_tp1.numpy()
    )


def _tf_example_to_tianshou_batch(tf_example: tf.train.Example) -> Batch:
    """Create a tianshou Batch replay sample from a TF example."""

    # Parse tf.Example.
    feature_description = {
        "o_t": tf.io.FixedLenFeature([4], tf.string),
        "o_tp1": tf.io.FixedLenFeature([4], tf.string),
        "a_t": tf.io.FixedLenFeature([], tf.int64),
        "a_tp1": tf.io.FixedLenFeature([], tf.int64),
        "r_t": tf.io.FixedLenFeature([], tf.float32),
        "d_t": tf.io.FixedLenFeature([], tf.float32),
        "episode_id": tf.io.FixedLenFeature([], tf.int64),
        "episode_return": tf.io.FixedLenFeature([], tf.float32),
    }
    data = tf.io.parse_single_example(tf_example, feature_description)

    # Process data.
    o_t = _decode_frames(data["o_t"])
    o_tp1 = _decode_frames(data["o_tp1"])
    a_t = tf.cast(data["a_t"], tf.int32)
    a_tp1 = tf.cast(data["a_tp1"], tf.int32)

    # Build tianshou Batch replay sample.
    return _make_tianshou_batch(o_t, a_t, data["r_t"], data["d_t"], o_tp1, a_tp1)


# Adapted From https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51
def download(url: str, fname: str, chunk_size=1024):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    if os.path.exists(fname):
        print(f"Found cached file at {fname}.")
        return
    with open(fname, 'wb') as ofile, tqdm(
        desc=fname,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = ofile.write(data)
            bar.update(size)


def process_shard(url: str, fname: str, ofname: str) -> None:
    download(url, fname)
    maxsize = 500000
    obs = np.ndarray((maxsize, 4, 84, 84), dtype="uint8")
    act = np.ndarray((maxsize, ), dtype="int64")
    rew = np.ndarray((maxsize, ), dtype="float32")
    done = np.ndarray((maxsize, ), dtype="bool")
    obs_next = np.ndarray((maxsize, 4, 84, 84), dtype="uint8")
    i = 0
    file_ds = tf.data.TFRecordDataset(fname, compression_type="GZIP")
    for example in file_ds:
        batch = _tf_example_to_tianshou_batch(example)
        obs[i], act[i], rew[i], done[i], obs_next[i] = (
            batch.obs, batch.act, batch.rew, batch.done, batch.obs_next
        )
        i += 1
        if i % 1000 == 0:
            print(f"...{i}", end="", flush=True)
    print("\nDataset size:", i)
    # Following D4RL dataset naming conventions
    with h5py.File(ofname, "w") as f:
        f.create_dataset("observations", data=obs, compression="gzip")
        f.create_dataset("actions", data=act, compression="gzip")
        f.create_dataset("rewards", data=rew, compression="gzip")
        f.create_dataset("terminals", data=done, compression="gzip")
        f.create_dataset("next_observations", data=obs_next, compression="gzip")


def process_dataset(
    task: str,
    download_path: str,
    dst_path: str,
    run_id: int = 1,
    shard_id: int = 0,
    total_num_shards: int = 100,
) -> None:
    fn = f"{task}/{_filename(run_id, shard_id, total_num_shards=total_num_shards)}"
    url = f"{URL_PREFIX}/{fn}"
    filepath = f"{download_path}/{fn}"
    ofname = f"{dst_path}/{fn}.hdf5"
    process_shard(url, filepath, ofname)


def main(args):
    if args.task not in ALL_GAMES:
        raise KeyError(f"`{args.task}` is not in the list of games.")
    fn = _filename(args.run_id, args.shard_id, total_num_shards=args.total_num_shards)
    dataset_path = os.path.join(args.dataset_dir, args.task, f"{fn}.hdf5")
    if os.path.exists(dataset_path):
        raise IOError(f"Found existing dataset at {dataset_path}. Will not overwrite.")
    args.cache_dir = os.environ.get("RLU_CACHE_DIR", args.cache_dir)
    args.dataset_dir = os.environ.get("RLU_DATASET_DIR", args.dataset_dir)
    cache_path = os.path.join(args.cache_dir, args.task)
    os.makedirs(cache_path, exist_ok=True)
    dst_path = os.path.join(args.dataset_dir, args.task)
    os.makedirs(dst_path, exist_ok=True)
    process_dataset(
        args.task,
        args.cache_dir,
        args.dataset_dir,
        run_id=args.run_id,
        shard_id=args.shard_id,
        total_num_shards=args.total_num_shards
    )


if __name__ == "__main__":
    parser = ArgumentParser(usage=__doc__)
    parser.add_argument("--task", required=True, help="Name of the Atari game.")
    parser.add_argument(
        "--run-id",
        type=int,
        default=1,
        help="Run id to download and convert. Value in [1..5]."
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="Shard id to download and convert. Value in [0..99]."
    )
    parser.add_argument(
        "--total-num-shards", type=int, default=100, help="Total number of shards."
    )
    parser.add_argument(
        "--dataset-dir",
        default=os.path.expanduser("~/.rl_unplugged/datasets"),
        help="Directory for converted hdf5 files.",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.path.expanduser("~/.rl_unplugged/cache"),
        help="Directory for downloaded original datasets.",
    )
    args = parser.parse_args()
    main(args)
