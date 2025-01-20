#!/bin/sh
#SBATCH --job-name=kv
#SBATCH --output=kv_%A.out
#SBATCH --error=kv_%A.err
#SBATCH --time=900
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jinfan@comp.nus.edu.sg
#SBATCH --gres=gpu:h100-47:1

. .venv/bin/activate
# srun python run_longbench.py --n_latest=48 >kv48.out 2>kv48.err
srun python run_longbench.py --n_latest=8 >kv48.out 2>kv48.err
# srun python run_longbench.py --n_latest=32 >kv32.out 2>kv32.err
