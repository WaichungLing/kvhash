#!/bin/sh
#SBATCH --job-name=kv-l16
#SBATCH --output=kv-l16_%A.out
#SBATCH --error=kv-l16_%A.err
#SBATCH --time=500
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jinfan@comp.nus.edu.sg
#SBATCH --gres=gpu:a100-80:1

# Run the training script
# add --enable_eviction if needed
srun python run_longbench.py \
    --enable_eviction \
    --model_name="meta-llama/Llama-3.2-3B-Instruct" \
    --recent_protect_budget=64 \
    --cache_budget=512 \
    --proxy_total=64 \
    --proxy_latest=16 \
    --n_recursion=-1 \
    --task="all" >kv.out 2>kv.err

# Run the evaluation script
# add --enable_eviction if needed
srun python eval_longbench.py \
    --enable_eviction \
    --model_name="meta-llama/Llama-3.2-3B-Instruct" \
    --recent_protect_budget=64 \
    --cache_budget=512 \
    --proxy_total=64 \
    --proxy_latest=16 \
    --n_recursion=-1 >ev.out 2>ev.err
