GPU_play=(0)
GPU_train=(3,4,5,6)
str_play='python play.py --data_path=./data/ --save_path=./go/ --game=go &'
str_train='python model.py &'
play_each_GPU=4

$str_train
echo 'Start training'
for gpu in $GPU
do 
export CUDA_VISIBLE_DEVICES=$gpu
for ((i=1;i<=$play_each_GPU;i++))
do
$str_play
done
done
