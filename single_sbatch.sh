#!/bin/sh
#SBATCH --job-name=kv
#SBATCH --output=kv_%A.out
#SBATCH --error=kv_%A.err
#SBATCH --time=600
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wling@comp.nus.edu.sg
#SBATCH --gres=gpu:a100-40:1

# Run the training script
srun python run_longbench.py \
    --model_name="meta-llama/Llama-3.2-3B-Instruct" \
    --cache_budget=512 \
    --proxy_total=64 \
    --proxy_latest=16 \
    --n_recursion=1 \
    --task="all" > kv.out 2> kv.err

# Run the evaluation script
srun python eval_longbench.py \
    --model_name="meta-llama/Llama-3.2-3B-Instruct" \
    --cache_budget=512 \
    --proxy_total=64 \
    --proxy_latest=16 \
    --n_recursion=1 > ev.out 2> ev.err
