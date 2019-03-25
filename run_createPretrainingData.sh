#!/bin/bash
#SBATCH --get-user-env
#SBATCH --job-name="BertDataPrep"
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --mem=200GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

python create_pretraining_data.py