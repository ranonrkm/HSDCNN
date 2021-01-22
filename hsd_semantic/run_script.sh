#!/bin/bash

#SBATCH -J hsd
#SBATCH -o out_hsd
#SBATCH -e err_hsd
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

export DATA_PATH='/scratch/16ee35016/data/cifar'

export IMPACT_PATH='/scratch/16ee35016/HSDCNN/hsd_semantic/impact_scores'

python -u main.py -H -p 1000 > test_1.out