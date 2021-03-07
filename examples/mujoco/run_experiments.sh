MAXSEED=10
LOGDIR="results"
for ((seed=0;seed<$MAXSEED;seed+=1))
do
    txtname=${1}_`date '+%m-%d-%H-%M-%S'`_seed_${seed}.txt
    python mujoco_sac.py \
    --task $1 \
    --epoch 200 \
    --seed $seed \
    --logdir $LOGDIR > $txtname 2>&1 &

    sleep 3s
done
echo "Experiments started."