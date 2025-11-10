Cluster-Oriented Pokémon Red RL (Actor–Learner)
================================================

Overview
--------
- Headless, efficient training (no live rendering).
- Multiple CPU actors generate experience and send completed episodes to a GPU learner via a prioritized recurrent replay buffer.
- Episodes are minimally logged (initial savestate bytes + action stream) for offline, full-length MP4 rendering after training.
- Unified model: conv + residual + optional spatial attention; single LSTM; dueling quantile head with NoisyLinear; auxiliary context reconstruction.
- Affordance-based action masking prevents invalid menu actions (e.g., LEFT/RIGHT in vertical lists).
- Per-map occupancy + saliency memory with local crop fed as auxiliary context.

Quick Start
-----------
1. Prepare ROM in this repository (e.g., `PokemonRedStreaming/pokemon_red.gb`).
2. Configure `cluster_config.json` or use defaults.
3. Launch training locally:

```bash
python -m cluster_rl.train_cluster --config cluster_rl/cluster_config.json
```

4. After training, generate videos:

```bash
python -m cluster_rl.replayer --rom PokemonRedStreaming/pokemon_red.gb --logs cluster_runs/<run_name>/episode_logs --out cluster_runs/<run_name>/videos --fps 30
```

Cluster Notes
-------------
- Use your university cluster interactive GPU job as documented in the Academic Compute Cluster guide (request a single GPU, sufficient CPU, disable rendering). See: Washington University Academic Compute Cluster.
- Training stops when either `target_total_steps` is reached or `max_runtime_seconds` expires (defaults to just under 4 hours). Submit a new job with the same `--run-name` to continue; progress is tracked in `cluster_runs/<run_name>/progress.json`.
- `online_params.pt` is refreshed after each target-network sync and at shutdown so actors (and subsequent jobs) pick up the latest weights.

Files
-----
- `train_cluster.py` — Orchestrates actors and learner.
- `actor.py` — Actor loop, affordances, episode logging for replayer.
- `learner.py` — R2D2-style learner with prioritized sequence replay.
- `model.py` — Unified architecture.
- `cluster_map_features.py` — Compact features (excludes HP) + occupancy crop.
- `occupancy_memory.py` — Wraps env with occupancy/saliency memory, savestate capture.
- `affordances.py` — UI signature and action masking memory.
- `replay_buffer.py` — Prioritized sequence replay.
- `replayer.py` — Offline full-episode MP4 rendering.
- `config.py` — Config dataclass and JSON loader.

Config
------
See `cluster_config.json` for tunables. Key entries:
- `target_total_steps`: total environment steps to accumulate (default 25 M).
- `replay_capacity_transitions`: replay buffer size in transitions (default 5 M; requires substantial RAM).
- `max_runtime_seconds`: soft wall-clock cutoff per job (default 13 500 s ≈ 3.75 h).
- `total_episodes`: optional actor cap (set 0 for unlimited episodes).


