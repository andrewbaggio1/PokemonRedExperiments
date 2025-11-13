from __future__ import annotations

import os
import queue
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.multiprocessing import Event, Queue

from .config import ClusterConfig
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
    online = ClusterDQN(obs_shape, context_dim, actions_count, use_spatial_attention=cfg.use_spatial_attention, lstm_hidden_size=cfg.lstm_hidden_size, num_quantiles=cfg.num_quantiles).to(device)
    maybe_load_state_dict(online, param_out_path)
    target = ClusterDQN(obs_shape, context_dim, actions_count, use_spatial_attention=cfg.use_spatial_attention, lstm_hidden_size=cfg.lstm_hidden_size, num_quantiles=cfg.num_quantiles).to(device)
    target.load_state_dict(online.state_dict())
    optimizer = optim.AdamW(online.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.999), weight_decay=1e-5)
    replay = PrioritizedSequenceReplay(capacity_transitions=cfg.replay_capacity_transitions, alpha=cfg.prioritized_alpha, beta_start=0.4, beta_frames=cfg.prioritized_beta_frames)
    env_steps_total = int(initial_env_steps)
    global_updates = 0
    last_sync = 0
    last_log_steps = env_steps_total
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
                save_json(progress_path, {"env_steps": env_steps_total})
                # Periodic progress log
                try:
                    if env_steps_total - last_log_steps >= max(1000, int(cfg.log_interval_steps)):
                        print(f"[learner] env_steps={env_steps_total:,} replay_size={len(replay):,} updates={global_updates:,}")
                        last_log_steps = env_steps_total
                except Exception:
                    pass
                if env_steps_total >= cfg.target_total_steps:
                    stop_event.set()
            if len(replay) >= cfg.learn_start_steps and not stop_event.is_set():
                seqs, weights, keys = replay.sample_sequences(cfg.batch_size, cfg.unroll_length, cfg.burn_in)
                (
                    obs_seq, ctx_seq, actions_seq, rewards_seq, discounts_seq, next_obs_seq, next_ctx_seq, dones_seq
                ) = _sequence_to_tensors(seqs, device)
                batch_size, T = actions_seq.shape[:2]
                # Burn-in LSTM state
                hidden = online.init_hidden(batch_size, device)
                with torch.no_grad():
                    online.reset_noise()
                    for t in range(cfg.burn_in):
                        _, hidden, _ = online(obs_seq[:, t, ...], ctx_seq[:, t, ...], hidden)
                # Unroll and compute losses
                online.reset_noise()
                target.reset_noise()
                quantiles_list = []
                aux_list = []
                act_list = []
                rew_list = []
                disc_list = []
                done_list = []
                next_quant_list = []
                h = hidden
                for t in range(cfg.burn_in, cfg.burn_in + cfg.unroll_length):
                    q_quant, h, aux = online(obs_seq[:, t, ...], ctx_seq[:, t, ...], h)
                    quantiles_list.append(q_quant)
                    aux_list.append(aux)
                    act_list.append(actions_seq[:, t, ...])
                    rew_list.append(rewards_seq[:, t, ...])
                    disc_list.append(discounts_seq[:, t, ...])
                    done_list.append(dones_seq[:, t, ...])
                # Next states for double Q
                with torch.no_grad():
                    h_t = hidden
                    for t in range(cfg.burn_in, cfg.burn_in + cfg.unroll_length):
                        q_next_online, h_t, _ = online(next_obs_seq[:, t, ...], next_ctx_seq[:, t, ...], h_t)
                        a_next = q_next_online.mean(dim=2).argmax(dim=1, keepdim=True).unsqueeze(-1)
                        q_next_tgt, _, _ = target(next_obs_seq[:, t, ...], next_ctx_seq[:, t, ...], None)
                        next_q = q_next_tgt.gather(1, a_next.expand(-1, -1, q_next_tgt.size(-1))).squeeze(1)
                        next_quant_list.append(next_q)
                # Stack
                quantiles = torch.stack(quantiles_list, dim=1)  # [B, T, A, K]
                aux_pred = torch.stack(aux_list, dim=1)         # [B, T, C]
                actions_b = torch.stack(act_list, dim=1).unsqueeze(-1).unsqueeze(-1)  # [B, T, 1, 1]
                chosen_quant = quantiles.gather(2, actions_b.expand(-1, -1, -1, quantiles.size(-1))).squeeze(2)  # [B,T,K]
                rewards_b = torch.stack(rew_list, dim=1).unsqueeze(-1)      # [B,T,1]
                discounts_b = torch.stack(disc_list, dim=1).unsqueeze(-1)   # [B,T,1]
                dones_b = torch.stack(done_list, dim=1).unsqueeze(-1)       # [B,T,1]
                next_chosen = torch.stack(next_quant_list, dim=1)           # [B,T,K]
                target_quant = rewards_b + discounts_b * (1.0 - dones_b) * next_chosen
                # Quantile Huber loss
                K = chosen_quant.size(-1)
                taus = (torch.arange(K, device=device, dtype=torch.float32) + 0.5) / K
                taus = taus.view(1, 1, K)
                diff = target_quant.unsqueeze(-2) - chosen_quant.unsqueeze(-1)  # [B,T,K,K]
                huber = torch.where(diff.abs() <= 1.0, 0.5 * diff.pow(2), diff.abs() - 0.5)
                quantile_loss = torch.abs(taus - (diff < 0).float()) * huber
                quantile_loss = quantile_loss.mean(dim=(2, 3))  # [B,T]
                # Auxiliary loss: reconstruct context (use same time slices)
                ctx_target = ctx_seq[:, cfg.burn_in : cfg.burn_in + cfg.unroll_length, ...]
                aux_pred = aux_pred  # already aligned with unroll
                aux_loss = F.mse_loss(aux_pred, ctx_target, reduction="none").mean(dim=2)  # [B,T]
                # Importance weights
                weights = torch.tensor(np.asarray(weights), dtype=torch.float32, device=device).view(-1, 1)  # [B,1]
                loss = (weights * (quantile_loss + 0.05 * aux_loss).mean(dim=1)).mean()
                optimizer.zero_grad(set_to_none=True)
                torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                loss.backward()
                optimizer.step()
                # Update priorities
                td_errors = (target_quant.mean(dim=2) - chosen_quant.mean(dim=2)).detach().abs().mean(dim=1).cpu().numpy()
                replay.update_priorities(keys, td_errors)
                global_updates += 1
                # Periodic target sync and param broadcast
                if global_updates - last_sync >= cfg.target_sync_interval:
                    target.load_state_dict(online.state_dict())
                    try:
                        tmp_path = f"{param_out_path}.tmp"
                        torch.save(online.state_dict(), tmp_path)
                        os.replace(tmp_path, param_out_path)
                    except Exception:
                        pass
                    last_sync = global_updates
                    try:
                        print(f"[learner] target synced at update={global_updates:,} env_steps={env_steps_total:,}")
                    except Exception:
                        pass
            if stop_event.is_set():
                break
    except KeyboardInterrupt:
        pass
    finally:
        try:
            tmp_path = f"{param_out_path}.tmp"
            torch.save(online.state_dict(), tmp_path)
            os.replace(tmp_path, param_out_path)
        except Exception:
            pass


