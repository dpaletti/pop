until taskset -c 0-14 python pop/main.py --run-file ~/pop_configs/run_files/env_rte/training/dpop/steps_1e5/dpop_rte_1e5_very_small.toml; do
           echo "Training crashed with exit code $?.  Resuming..." >&2
       sleep 1
done       
