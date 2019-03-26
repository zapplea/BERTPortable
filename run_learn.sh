#!/bin/bash
#SBATCH --get-user-env
#SBATCH --job-name="BERT"
#SBATCH --time=140:00:00
#SBATCH --nodes=1
#SBATCH --mem=200GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
echo "loading"
module load python/3.6.1
module load tensorflow/1.12.0-py36-gpu
echo "loaded"

python run_learn.sh --gpu_num 2