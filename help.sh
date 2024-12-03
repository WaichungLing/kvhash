for i in $(seq 0 0.1 1); do
  cmd="srun --gres=gpu:a100-80:1 --time=300 python run_longbench.py --cache_budget=$i"
  echo "Running $cmd"
  # $cmd
done
