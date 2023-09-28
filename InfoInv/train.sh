#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 24:00:00
#SBATCH -o /dev/null
#SBATCH --gres gpu:1

#echo "CUDA versions are stored here:"
#find /usr/lib/cuda* -maxdepth 0
#echo "Default GCC:"
#gcc --version

python3 main.py --config configs/lego.txt