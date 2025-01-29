#!/bin/sh
#SBATCH --job-name=kvr-2
#SBATCH --output=kv-2_%A.out
#SBATCH --error=kv-2_%A.err
#SBATCH --time=1000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jinfan@comp.nus.edu.sg
#SBATCH --gres=gpu:a100-80:1

. .venv/bin/activate

# Run the training script
# add --enable_eviction if needed
srun python run_longbench.py \
    --enable_eviction \
    --model_name="meta-llama/Llama-3.2-3B-Instruct" \
    --recent_protect_budget=32 \
    --cache_budget=512 \
    --proxy_total=64 \
    --proxy_latest=16 \
    --n_recursion=2 \
    --task="all" || exit 1

# Run the evaluation script
# add --enable_eviction if needed
srun python eval_longbench.py \
    --enable_eviction \
    --model_name="meta-llama/Llama-3.2-3B-Instruct" \
    --recent_protect_budget=32 \
    --cache_budget=512 \
    --proxy_total=64 \
    --proxy_latest=16 \
    --n_recursion=2 || exit 1
