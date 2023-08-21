#!/bin/bash
#SBATCH -p gpu22
#SBATCH -t 16:00:00
#SBATCH -o /dev/null
#SBATCH --gres gpu:1

#echo "CUDA versions are stored here:"
#find /usr/lib/cuda* -maxdepth 0
#echo "Default GCC:"
#gcc --version

[ -z $1 ] && exit 1

name="${1}"
dataset_name='dtu'
data_root="/CT/Neural-Gauge-Fields/work/Datasets/DTU/scan${1}/trainData"

random_sample='balanced'
random_sample_size=24
sample_num=64
primitive_type='square'
points_per_primitive=2500

loss_color_weight=1
loss_bg_weight=1
loss_inverse_mapping_weight=0

# training
batch_size=1
lr=0.0001
gpu_ids='0'
checkpoints_dir='./checkpoints/'
resume_checkpoints_dir='./checkpoints'
save_iter_freq=5000
niter=500000
niter_decay=0
n_threads=0
train_and_test=1
test_num=1
print_freq=20
test_freq=10000


python3 train.py  \
        --name=$name  \
        --dataset_name=$dataset_name  \
        --data_root=$data_root  \
        --random_sample=$random_sample  \
        --random_sample_size=$random_sample_size  \
        --sample_num=$sample_num  \
        --primitive_type=$primitive_type  \
        --points_per_primitive=$points_per_primitive  \
        --loss_color_weight=$loss_color_weight  \
        --loss_bg_weight=$loss_bg_weight  \
        --loss_inverse_mapping_weight=$loss_inverse_mapping_weight  \
        --batch_size=$batch_size  \
        --lr=$lr  \
        --gpu_ids=$gpu_ids  \
        --checkpoints_dir=$checkpoints_dir  \
        --save_iter_freq=$save_iter_freq  \
        --niter=$niter  \
        --niter_decay=$niter_decay  \
        --n_threads=$n_threads  \
        --train_and_test=$train_and_test  \
        --test_num=$test_num  \
        --print_freq=$print_freq  \
        --test_freq=$test_freq  \
#        --resume_dir=$resume_checkpoints_dir/${1}
#        --resume_epoch=400000  \
#        --resume_dir=$checkpoints_dir  \

