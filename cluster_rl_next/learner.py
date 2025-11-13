from __future__ import annotations

import os
import queue
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from torch.multiprocessing import Event, Queue
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import ClusterConfig
from .checkpointing import prune_checkpoints, save_checkpoint, load_checkpoint
from .model import ClusterDQN
from .replay_buffer import PrioritizedSequenceReplay, Transition
from .utils import ensure_dir, maybe_load_state_dict, save_json


def _sequence_to_tensors(seqs: Sequence[Sequence[Transition]], device: torch.device):
    # seqs: list of (burn_in + unroll) transitions per batch item
    batch = len(seqs)
    T = len(seqs[0])
    # Build tensors
    obs_np = np.stack([t.obs for s in seqs for t in s])
    next_obs_np = np.stack([t.next_obs for s in seqs for t in s])
    # Decompress if stored as uint8
    if obs_np.dtype == np.uint8:
        obs_np = obs_np.astype(np.float32) / 255.0
    if next_obs_np.dtype == np.uint8:
        next_obs_np = next_obs_np.astype(np.float32) / 255.0
    ctx_np = np.stack([t.context for s in seqs for t in s]).astype(np.float32, copy=False)
    next_ctx_np = np.stack([t.next_context for s in seqs for t in s]).astype(np.float32, copy=False)
    obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
    ctx = torch.tensor(ctx_np, dtype=torch.float32, device=device)
    next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
    next_ctx = torch.tensor(next_ctx_np, dtype=torch.float32, device=device)
    actions = torch.tensor([t.action for s in seqs for t in s], dtype=torch.long, device=device)
    rewards = torch.tensor([t.reward for s in seqs for t in s], dtype=torch.float32, device=device)
    discounts = torch.tensor([t.discount for s in seqs for t in s], dtype=torch.float32, device=device)
    dones = torch.tensor([t.done for s in seqs for t in s], dtype=torch.float32, device=device)
    # Reshape to [batch, T, ...]
    def rs(x):
        return x.view(batch, T, *x.shape[1:])
    return (
        rs(obs),
        rs(ctx),
        rs(actions),
        rs(rewards),
        rs(discounts),
        rs(next_obs),
        rs(next_ctx),
        rs(dones),
    )


def learner_loop(
    cfg: ClusterConfig,
    param_out_path: str,
    checkpoint_dir: str,
    resume_checkpoint: Optional[str],
    in_queue: Queue,
    device: torch.device,
    obs_shape: Tuple[int, int, int],
    context_dim: int,
    stop_event: Event,
    progress_path: str,
    initial_env_steps: int,
    n_actions: int,
) -> None:
    """
    Learner loop:
      - consumes completed episodes from in_queue
      - fills prioritized sequence replay
      - optimizes ClusterDQN with distributional dueling + auxiliary loss
      - periodically writes params to param_out_path for actors to read
    """
    ensure_dir(os.path.dirname(param_out_path))
    ensure_dir(os.path.dirname(progress_path))
    actions_count = max(1, int(n_actions))
    online = ClusterDQN(obs_shape, context_dim, actions_count, cfg=cfg).to(device)
    target = ClusterDQN(obs_shape, context_dim, actions_count, cfg=cfg).to(device)
    opt_cfg = cfg.training.optimizer
    optimizer = optim.AdamW(
        online.parameters(),
        lr=opt_cfg.lr,
        betas=opt_cfg.betas,
        eps=opt_cfg.eps,
        weight_decay=opt_cfg.weight_decay,
    )
    scaler = GradScaler(enabled=bool(cfg.training.amp) and device.type == "cuda")
    replay = PrioritizedSequenceReplay(
        capacity_transitions=cfg.replay_capacity_transitions,
        alpha=cfg.prioritized_alpha,
        beta_start=0.4,
        beta_frames=cfg.prioritized_beta_frames,
    )
    env_steps_total = int(initial_env_steps)
    global_updates = 0
    last_sync = 0
    last_log_steps = env_steps_total
    ckpt_interval = max(1, cfg.checkpointing.interval_minutes) * 60
    last_ckpt_time = time.monotonic()
    ensure_dir(checkpoint_dir)

    # -------------------------------
    # Optional Distributed setup (DDP)
    # -------------------------------
    ddp_active = False
    rank = 0
    world_size = 1
    store_url = os.environ.get("DDP_STORE")
    try:
        env_world = int(os.environ.get("WORLD_SIZE", "1"))
        env_rank = int(os.environ.get("RANK", "0"))
    except Exception:
        env_world = 1
        env_rank = 0
    if env_world > 1 and store_url:
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend, init_method=store_url, world_size=env_world, rank=env_rank)
        ddp_active = True
        rank = env_rank
        world_size = env_world
        if device.type == "cuda":
            torch.cuda.set_device(0)
        # Wrap online model; target remains local
        online = DDP(online, device_ids=[0] if device.type == "cuda" else None, output_device=0 if device.type == "cuda" else None, find_unused_parameters=False)

    def _is_primary() -> bool:
        return (not ddp_active) or (rank == 0)

    def _model_for_saving(module: torch.nn.Module) -> torch.nn.Module:
        return module.module if hasattr(module, "module") else module
    def _online_module() -> torch.nn.Module:
        return _model_for_saving(online)

    def _write_param_snapshot():
        if not _is_primary():
            return
        try:
            tmp_path = f"{param_out_path}.tmp"
            torch.save(_model_for_saving(online).state_dict(), tmp_path)
            os.replace(tmp_path, param_out_path)
        except Exception:
            pass

    def _save_checkpoint(force: bool = False, suffix: Optional[str] = None):
        nonlocal last_ckpt_time
        if not _is_primary():
            return
        if not force and (time.monotonic() - last_ckpt_time) < ckpt_interval:
            return
        metadata = {
            "env_steps": env_steps_total,
            "global_updates": global_updates,
            "last_sync": last_sync,
            "timestamp": time.time(),
        }
        replay_state = replay.state_dict() if cfg.checkpointing.include_replay_stats else None
        filename = suffix or f"ckpt_{env_steps_total:012d}.pt"
        path = os.path.join(checkpoint_dir, filename)
        save_checkpoint(path, _model_for_saving(online), optimizer, replay_state if cfg.checkpointing.include_replay_stats else None, metadata)
        prune_checkpoints(checkpoint_dir, cfg.checkpointing.max_keep)
        last_ckpt_time = time.monotonic()

    # Resume if checkpoint provided
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        try:
            payload = load_checkpoint(resume_checkpoint, _model_for_saving(online), optimizer)
            metadata = payload.get("metadata", {})
            env_steps_total = max(env_steps_total, int(metadata.get("env_steps", env_steps_total)))
            global_updates = int(metadata.get("global_updates", global_updates))
            last_sync = int(metadata.get("last_sync", last_sync))
            if cfg.checkpointing.include_replay_stats and "replay" in payload:
                replay.load_state_dict(payload["replay"])
            target.load_state_dict(_model_for_saving(online).state_dict())
            _write_param_snapshot()
            if _is_primary():
                print(f"[learner] resumed from checkpoint {resume_checkpoint}")
        except Exception as exc:
            if _is_primary():
                print(f"[learner] failed to resume from checkpoint {resume_checkpoint}: {exc}")
            target.load_state_dict(_model_for_saving(online).state_dict())
    else:
        maybe_load_state_dict(_model_for_saving(online), param_out_path)
        target.load_state_dict(_model_for_saving(online).state_dict())
        _write_param_snapshot()

    # Barrier to synchronize start if using DDP (ensures all ranks are initialized)
    if ddp_active:
        dist.barrier()

    ddp_warmup_synced = not ddp_active

    try:
        while not stop_event.is_set():
            # Consume new episodes
            try:
                typ, payload = in_queue.get(timeout=1.0)
            except queue.Empty:
                typ = None
            if typ == "episode":
                trans: List[Transition] = []
                for (obs, ctx, action, reward, discount, next_obs, next_ctx, done) in payload:
                    trans.append(Transition(obs, ctx, int(action), float(reward), float(discount), next_obs, next_ctx, bool(done)))
                replay.add_episode(trans)
                env_steps_total += len(trans)
                if _is_primary():
                    save_json(progress_path, {"env_steps": env_steps_total, "updates": global_updates})
                # Periodic progress log
                try:
                    if _is_primary() and env_steps_total - last_log_steps >= max(1000, int(cfg.log_interval_steps)):
                        print(f"[learner] env_steps={env_steps_total:,} replay_size={len(replay):,} updates={global_updates:,}")
                        last_log_steps = env_steps_total
                except Exception:
                    pass
                if env_steps_total >= cfg.target_total_steps:
                    stop_event.set()
                # Synchronize warm-up across ranks before starting training
                if ddp_active and not ddp_warmup_synced:
                    try:
                        ready_flag = torch.tensor([1 if len(replay) >= cfg.learn_start_steps else 0], device=device if device.type == "cuda" else torch.device("cpu"))
                        if device.type != "cuda":
                            ready_flag = ready_flag.to(torch.device("cpu"))
                        dist.all_reduce(ready_flag, op=dist.ReduceOp.SUM)
                        if int(ready_flag.item()) >= world_size:
                            ddp_warmup_synced = True
                            if _is_primary():
                                print("[learner] DDP warm-up synchronized across all ranks; starting training.")
                    except Exception:
                        # If reduction fails, bail out to avoid deadlock
                        stop_event.set()
                _save_checkpoint()
            if len(replay) >= cfg.learn_start_steps and not stop_event.is_set() and (ddp_warmup_synced or not ddp_active):
                seqs, weights, keys = replay.sample_sequences(cfg.batch_size, cfg.unroll_length, cfg.burn_in)
                (
                    obs_seq, ctx_seq, actions_seq, rewards_seq, discounts_seq, next_obs_seq, next_ctx_seq, dones_seq
                ) = _sequence_to_tensors(seqs, device)
                batch_size, T = actions_seq.shape[:2]
                # Burn-in LSTM state
                hidden = _online_module().init_hidden(batch_size, device)
                with torch.no_grad():
                    _online_module().reset_noise()
                    for t in range(cfg.burn_in):
                        _, hidden, _ = online(obs_seq[:, t, ...], ctx_seq[:, t, ...], hidden)
                # Unroll and compute losses
                _online_module().reset_noise()
                target.reset_noise()
                quantiles_list = []
                aux_buffers: dict[str, List[torch.Tensor]] = {}
                act_list = []
                rew_list = []
                disc_list = []
                done_list = []
                next_quant_list = []
                h = hidden
                amp_enabled = scaler.is_enabled()
                for t in range(cfg.burn_in, cfg.burn_in + cfg.unroll_length):
                    with autocast(enabled=amp_enabled):
                        q_quant, h, aux = online(obs_seq[:, t, ...], ctx_seq[:, t, ...], h)
                    quantiles_list.append(q_quant)
                    if isinstance(aux, dict):
                        for key, val in aux.items():
                            aux_buffers.setdefault(key, []).append(val)
                    else:
                        aux_buffers.setdefault("context_recon", []).append(aux)
                    act_list.append(actions_seq[:, t, ...])
                    rew_list.append(rewards_seq[:, t, ...])
                    disc_list.append(discounts_seq[:, t, ...])
                    done_list.append(dones_seq[:, t, ...])
                with torch.no_grad():
                    h_t = hidden
                    for t in range(cfg.burn_in, cfg.burn_in + cfg.unroll_length):
                        with autocast(enabled=amp_enabled):
                            q_next_online, h_t, _ = online(next_obs_seq[:, t, ...], next_ctx_seq[:, t, ...], h_t)
                            a_next = q_next_online.mean(dim=2).argmax(dim=1, keepdim=True).unsqueeze(-1)
                            q_next_tgt, _, _ = target(next_obs_seq[:, t, ...], next_ctx_seq[:, t, ...], None)
                            next_q = q_next_tgt.gather(1, a_next.expand(-1, -1, q_next_tgt.size(-1))).squeeze(1)
                        next_quant_list.append(next_q)
                quantiles = torch.stack(quantiles_list, dim=1)
                actions_stack = torch.stack(act_list, dim=1)
                actions_b = actions_stack.unsqueeze(-1).unsqueeze(-1)
                chosen_quant = quantiles.gather(2, actions_b.expand(-1, -1, -1, quantiles.size(-1))).squeeze(2)
                rewards_b = torch.stack(rew_list, dim=1).unsqueeze(-1)
                discounts_b = torch.stack(disc_list, dim=1).unsqueeze(-1)
                dones_b = torch.stack(done_list, dim=1).unsqueeze(-1)
                next_chosen = torch.stack(next_quant_list, dim=1)
                target_quant = rewards_b + discounts_b * (1.0 - dones_b) * next_chosen
                aux_preds = {k: torch.stack(v, dim=1) for k, v in aux_buffers.items()}
                ctx_target = ctx_seq[:, cfg.burn_in : cfg.burn_in + cfg.unroll_length, ...]
                weights_tensor = torch.tensor(np.asarray(weights), dtype=torch.float32, device=device).view(-1, 1)
                aux_cfg = getattr(cfg, "heads", None)
                aux_weights = getattr(aux_cfg, "aux", None) if aux_cfg is not None else None
                clip_val = float(getattr(cfg.training, "grad_clip", 0.0))
                with autocast(enabled=amp_enabled):
                    K = chosen_quant.size(-1)
                    taus = (torch.arange(K, device=device, dtype=torch.float32) + 0.5) / K
                    taus = taus.view(1, 1, K)
                    diff = target_quant.unsqueeze(-2) - chosen_quant.unsqueeze(-1)
                    huber = torch.where(diff.abs() <= 1.0, 0.5 * diff.pow(2), diff.abs() - 0.5)
                    quantile_loss = torch.abs(taus - (diff < 0).float()) * huber
                    quantile_loss = quantile_loss.mean(dim=(2, 3))
                    aux_terms = []
                    if aux_weights and aux_weights.context_recon_weight > 0 and "context_recon" in aux_preds:
                        ctx_pred = aux_preds["context_recon"]
                        ctx_err = F.mse_loss(ctx_pred, ctx_target, reduction="none").mean(dim=2)
                        aux_terms.append(aux_weights.context_recon_weight * ctx_err)
                    if aux_weights and aux_weights.inverse_dynamics_weight > 0 and "inverse_dynamics" in aux_preds:
                        inv_pred = aux_preds["inverse_dynamics"].contiguous()
                        inv_loss = F.cross_entropy(
                            inv_pred.view(-1, inv_pred.size(-1)),
                            actions_stack.view(-1),
                            reduction="none",
                        ).view(batch_size, cfg.unroll_length)
                        aux_terms.append(aux_weights.inverse_dynamics_weight * inv_loss)
                    if aux_weights and aux_weights.reward_forecast_weight > 0 and "reward_forecast" in aux_preds:
                        rew_pred = aux_preds["reward_forecast"].squeeze(-1)
                        rew_err = F.mse_loss(rew_pred, rewards_b.squeeze(-1), reduction="none")
                        aux_terms.append(aux_weights.reward_forecast_weight * rew_err)
                    aux_loss = sum(aux_terms) if aux_terms else None
                    total_loss = quantile_loss
                    if aux_loss is not None:
                        total_loss = total_loss + aux_loss
                    loss = (weights_tensor * total_loss.mean(dim=1)).mean()
                optimizer.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(online.parameters(), clip_val)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(online.parameters(), clip_val)
                    optimizer.step()
                # Update priorities
                td_errors = (
                    target_quant.float().mean(dim=2) - chosen_quant.float().mean(dim=2)
                ).detach().abs().mean(dim=1).cpu().numpy()
                replay.update_priorities(keys, td_errors)
                global_updates += 1
                # Periodic target sync and param broadcast
                if global_updates - last_sync >= cfg.target_sync_interval:
                    target.load_state_dict(_model_for_saving(online).state_dict())
                    _write_param_snapshot()
                    last_sync = global_updates
                    try:
                        if _is_primary():
                            print(f"[learner] target synced at update={global_updates:,} env_steps={env_steps_total:,}")
                    except Exception:
                        pass
            if stop_event.is_set():
                break
    except KeyboardInterrupt:
        pass
    finally:
        _write_param_snapshot()
        _save_checkpoint(force=True, suffix=f"ckpt_final_{env_steps_total:012d}.pt")
        if ddp_active:
            try:
                dist.barrier()
                dist.destroy_process_group()
            except Exception:
                pass
