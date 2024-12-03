for i in $(seq 0.2 0.2 1); do
  cmd="srun --gres=gpu:a100-80:1 -o $i.log --time=300 python run_longbench.py --cache_budget=$i"
  echo "Running $cmd"
  $cmd >$i-cmd.log &
done
