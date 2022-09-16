until python -m debugpy --listen localhost:5678 --wait-for-client pop/main.py --run-file ~/pop_configs/run_files/env_rte/evaluation/dpop_rte_load.toml; do
    echo "Training crashed with exit code $?.  Resuming..." >&2
    sleep 1
done
