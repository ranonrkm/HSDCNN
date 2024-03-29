#!/bin/bash

#SBATCH -J hsd
#SBATCH -o out_hsd
#SBATCH -e err_hsd20
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=ranajoy@iitkgp.ac.in
#SBATCH --mail-type=ALL

module load compiler/cuda/10.1

source activate /home/16ee35016/anaconda2/envs/HSD
cd $SCRATCH
cd HSDCNN/hsd_semantic

#python -u check_clusters.py > test.out
python -u main.py -p 10000 --cfg config/cifar10_10.yaml > test_10_10.out
#python -u test_impact.py --cfg config/cifar100_XE.yaml > test_100_XE.out
#python -u main.py --pretrained -H -p 10000 --cfg config/cifar100_XE.yaml > test_100_XE.out
