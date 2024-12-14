#!/bin/sh
#SBATCH --job-name=kv
#SBATCH --output=kv_%A.out
#SBATCH --error=kv_%A.err
#SBATCH --time=600
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jinfan@comp.nus.edu.sg
#SBATCH --gres=gpu:a100-80:1

. .venv/bin/activate

run() {
  srun python run_longbench.py --cache_budget=$1 >kv_$1.out 2>kv_$1.err
}

eva() {
  srun python eval_longbench.py --cache_budget=$1 >ev_$1.out 2>ev_$1.err
}

run 1.0
eva 1.0

run 0.8
eva 0.8

run 0.6
eva 0.6

run 0.4
eva 0.4

run 0.2
eva 0.2
