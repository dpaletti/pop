until taskset -c 0-14 python pop/main.py --run-file ~/pop_configs/run_files/env_rte/training/dpop/dpop_rte_curiosity.toml; do
           echo "Training crashed with exit code $?.  Resuming..." >&2
       sleep 1
done       
