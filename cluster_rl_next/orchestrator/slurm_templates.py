from __future__ import annotations

from textwrap import dedent


def render_train_job(
    *,
    config_path: str,
    run_name: str,
    gpus: int = 1,
    mem_per_gpu: str = "20G",
    hours: int = 4,
    partition: str = "gpu-linuxlab",
    rank: int | None = None,
    world_size: int | None = None,
    ddp_store_path: str | None = None,
) -> str:
    time_str = f"{hours:02d}:00:00"
    python_bin = "/project/scratch01/compiling/a.a.baggio/PokemonRedExperiments/.venv/bin/python"
    ddp_block = ""
    if world_size is not None and world_size > 1:
        rank_val = 0 if rank is None else int(rank)
        store = ddp_store_path or f"cluster_runs/{run_name}/ddp_store"
        ddp_block = f"""
    # DDP environment
    export WORLD_SIZE={int(world_size)}
    export RANK={rank_val}
    export LOCAL_RANK=0
    mkdir -p "$(dirname "$PWD/{store}")"
    export DDP_STORE="file://$PWD/{store}"
    """
    script = f"""
    #!/bin/bash
    #SBATCH -p {partition}
    #SBATCH -A engr-class-any
    #SBATCH -J cluster_rl_{run_name}
    #SBATCH -c 8
    #SBATCH --gres=gpu:{gpus}
    #SBATCH --mem-per-gpu={mem_per_gpu}
    #SBATCH --time={time_str}
    #SBATCH --output=cluster_runs/{run_name}/logs/slurm-%j.out
    #SBATCH --error=cluster_runs/{run_name}/logs/slurm-%j.err

    mkdir -p "cluster_runs/{run_name}/logs"
    mkdir -p "cluster_runs/{run_name}/checkpoints"
    export OMP_NUM_THREADS=8
    {ddp_block}
    echo "Starting ClusterRL Next training run {run_name} at $(date)"
    {python_bin} -m cluster_rl_next.train_cluster --config {config_path}
    """
    return dedent(script).strip() + "\n"
