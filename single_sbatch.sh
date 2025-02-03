#!/bin/sh
#SBATCH --job-name=kv-3232
#SBATCH --output=kv-3232_%A.out
#SBATCH --error=kv-3232_%A.err
#SBATCH --time=100
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jinfan@comp.nus.edu.sg
#SBATCH --gres=gpu:a100-80:1

# 3232 = Latest 32, PCA 32

. .venv/bin/activate

# Run the training script
# add --enable_eviction if needed
srun python run_longbench.py \
    --enable_eviction \
    --model_name="meta-llama/Llama-3.2-3B-Instruct" \
    --recent_protect_budget=64 \
    --cache_budget=512 \
    --proxy_total=64 \
    --proxy_latest=32 \
    --n_recursion=-1 \
    --task="all"

# Run the evaluation script
# add --enable_eviction if needed
srun python eval_longbench.py \
    --enable_eviction \
    --model_name="meta-llama/Llama-3.2-3B-Instruct" \
    --recent_protect_budget=64 \
    --cache_budget=512 \
    --proxy_total=64 \
    --proxy_latest=32 \
    --n_recursion=-1
