for i in $(seq 0.2 0.2 1); do
  cmd="srun python eval_longbench.py --cache_budget=$i"
  echo "Running $cmd"
  $cmd >$i-cmd.log &
done
