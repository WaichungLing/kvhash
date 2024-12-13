#!/bin/sh
#SBATCH --job-name=kv-eva
#SBATCH --output=kv-eva_%A.out
#SBATCH --error=kv-eva_%A.err
#SBATCH --time=30
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jinfan@comp.nus.edu.sg

. .venv/bin/activate
srun python eval_longbench.py --cache_budget=0.4
