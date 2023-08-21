#!/bin/bash
[ -z $1 ] && exit 1

name="${1}"
dataset_name='dtu'
data_root="/CT/Neural-Gauge-Fields/work/Datasets/DTU/scan${1}/trainData"
target_texture='None'
#target_texture='/CT/Neural-Gauge-Fields/work/NERM/NeuTex/data/texture10.png'

random_sample='balanced'
random_sample_size=32
[ `whoami` = 'root' ] && random_sample_size=24
sample_num=64
[ `whoami` = 'root' ] && sample_num=256
primitive_type='square'
points_per_primitive=2500
gpu_ids='0'
checkpoints_dir='./checkpoints/'
resume_checkpoints_dir='./checkpoints'

srun -p gpu20 -t 45:00 --gres gpu:1 python3 visualize.py  \
        --name=$name  \
        --dataset_name=$dataset_name  \
        --data_root=$data_root  \
        --random_sample=$random_sample  \
        --random_sample_size=$random_sample_size  \
        --sample_num=$sample_num  \
        --primitive_type=$primitive_type  \
        --points_per_primitive=$points_per_primitive  \
        --gpu_ids=$gpu_ids  \
        --checkpoints_dir=$checkpoints_dir  \
        --target_texture=$target_texture  \
        --resume_dir=$resume_checkpoints_dir/${1}
