#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python mujoco_td3.py \
--task "HalfCheetah-v3" \
--logdir "log" > 1.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0 python mujoco_td3.py \
--task "Hopper-v3" \
--logdir "log" > 2.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python mujoco_td3.py \
--task "Walker2d-v3" \
--logdir "log" > 3.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python mujoco_td3.py \
--task "Ant-v3" \
--logdir "log" > 4.txt 2>&1 &

CUDA_VISIBLE_DEVICES=2 python mujoco_td3.py \
--task "Humanoid-v3" \
--logdir "log" > 5.txt 2>&1 &

CUDA_VISIBLE_DEVICES=2 python mujoco_td3.py \
--task "InvertedPendulum-v2" \
--logdir "log" \
--start-timesteps 1000 > 6.txt 2>&1 &

CUDA_VISIBLE_DEVICES=3 python mujoco_td3.py \
--task "InvertedDoublePendulum-v2" \
--logdir "log" \
--start-timesteps 1000 > 7.txt 2>&1 &

CUDA_VISIBLE_DEVICES=3 python mujoco_td3.py \
--task "Reacher-v2" \
--logdir "log" \
--start-timesteps 1000 > 8.txt 2>&1 &

