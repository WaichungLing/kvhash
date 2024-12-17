#!/bin/sh
#SBATCH --job-name=kv
#SBATCH --output=kv_%A.out
#SBATCH --error=kv_%A.err
#SBATCH --time=600
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jinfan@comp.nus.edu.sg
#SBATCH --gres=gpu:a100-80:1

. .venv/bin/activate

srun python run_longbench.py --enable_kvhash=False >kv.out 2>kv.err

srun python eval_longbench.py >ev.out 2>ev.err
