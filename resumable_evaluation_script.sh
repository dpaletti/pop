until python pop/main.py --run-file ~/pop_configs/run_files/env_rte/evaluation/dpop_rte_load.toml; do
    echo "Evaluation crashed with exit code $?.  Trying Again..." >&2
    sleep 1
done
