#!/bin/bash

LOGDIR="results"
TASK=$1

echo "Experiments started."
for seed in $(seq 1 10)
do
    python mujoco_sac.py --task $TASK --epoch 200 --seed $seed --logdir $LOGDIR > ${TASK}_`date '+%m-%d-%H-%M-%S'`_seed_$seed.txt 2>&1
done
