#!/bin/sh
#SBATCH --job-name=kv
#SBATCH --output=kv_%A.out
#SBATCH --error=kv_%A.err
#SBATCH --time=600
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wling@comp.nus.edu.sg
#SBATCH --gres=gpu:a100-40:1

srun python run_longbench.py >kv.out 2>kv.err
