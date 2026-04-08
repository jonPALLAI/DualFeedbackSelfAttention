#!/usr/bin/env python3

"""
Distributional RL with shared CNN patch encoder and single-step recurrent Transformer.
Hyperparameters load from config/training_config.json (override with --config).
"""

import argparse
import os
import sys
import numpy as np
import torch

_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE)
sys.path.insert(0, os.path.join(_BASE, 'Agent', 'DistActCritAgent'))
sys.path.insert(0, os.path.join(_BASE, 'Agent', 'Model'))

from config_loader import load_training_config, token_dims_from_model
from checkpoint_utils import bind_latest_checkpoint_paths, resolve_checkpoint_dir
from device_utils import device_name, get_torch_device
from OCDEnv import ChangeDetectionEnv
from agent import Agent as AgentPlanner
from VWMNET import TransformerNetwork


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train distributional RL agent (config-driven).')
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to JSON config (defaults to config/training_config.json next to this script).',
    )
    args = parser.parse_args()

    cfg = load_training_config(args.config)
    run = cfg['run']
    env_cfg = cfg['environment']
    m = cfg['model']
    t = cfg['training']

    mod_num = int(run['mod_num'])
    n_games = int(run['n_games'])
    chkpt_dir = resolve_checkpoint_dir(run['checkpoint_dir'], _BASE)
    os.makedirs(chkpt_dir, exist_ok=True)
    print(f"Checkpoint directory: {chkpt_dir}")

    token_dim, state_dim = token_dims_from_model(m)
    embed_dim = int(m['embed_dim'])
    spatial_dim = int(m['spatial_encoding_dim'])
    temporal_dim = int(m['temporal_encoding_dim'])
    patch_length = int(m['patch_length'])
    dim_recurrent = int(m['dim_recurrent'])
    assert patch_length == 4

    device = get_torch_device()
    print(f"Using device: {device} ({device_name(device)})")

    rec_vit_lr = float(t.get('rec_vit_lr', m.get('transformer_lr', 1e-5)))
    actor_lr = float(t.get('actor_lr', t.get('alpha', 1e-4)))
    critic_lr = float(t.get('critic_lr', t.get('beta', 1e-4)))
    print(
        f"Learning rates: rec_vit_lr={rec_vit_lr:g}, actor_lr={actor_lr:g}, critic_lr={critic_lr:g}"
    )

    transformer = TransformerNetwork(
        rec_vit_lr=rec_vit_lr,
        embed_dim=embed_dim,
        spatial_dim=spatial_dim,
        temporal_dim=temporal_dim,
        patch_encoder_name='patch_encoder' + str(mod_num),
        chkpt_dir=chkpt_dir,
        hidden_dim=int(m['transformer_hidden_dim']),
        ffn_dim=int(m['transformer_ffn_dim']),
        n_actions=2,
        name='transformer1' + str(mod_num),
        patch_length=patch_length,
        dim_recurrent=dim_recurrent,
        dropout=float(m['transformer_dropout']),
        embedding_noise_std=float(m['transformer_embedding_noise_std']),
    ).to(device)
    transformer.train()

    _ms = t.get('max_seq_len')
    max_seq_len = None if (_ms is None or _ms == 0) else int(_ms)

    agent = AgentPlanner(
        alpha=actor_lr,
        beta=critic_lr,
        input_dims=state_dim,
        tau=float(t['tau']),
        env=None,
        gamma=float(t['gamma']),
        warmup=int(t['warmup']),
        n_actions=2,
        max_size=int(t['buffer_max_size']),
        batch_size=int(t['batch_size']),
        exploration_noise=float(t['exploration_noise']),
        name=str(mod_num),
        encoder=transformer.patch_encoder,
        max_seq_len=max_seq_len,
        num_particles=int(m['num_particles']),
        dim_recurrent=dim_recurrent,
        actor_hidden_dim=int(m['actor_hidden_dim']),
        actor_fc1_dims=int(m['actor_fc1_dims']),
        actor_fc2_dims=int(m['actor_fc2_dims']),
        critic_hidden_dim=int(m['critic_hidden_dim']),
        critic_fc1_dims=int(m['critic_fc1_dims']),
        critic_fc2_dims=int(m['critic_fc2_dims']),
        eta=float(t['eta']),
        entropy_coef=float(t['entropy_coef']),
        grad_clip_max_norm=float(t['grad_clip_max_norm']),
        token_dim=token_dim,
        embed_dim=embed_dim,
        V_max=float(m['V_max']),
        patch_length=patch_length,
        replay_buffer_alpha=float(t['replay_buffer_alpha']),
        replay_buffer_sample_beta=float(t['replay_buffer_sample_beta']),
        chkpt_dir=chkpt_dir,
    )

    env = ChangeDetectionEnv(
        theta=float(env_cfg['theta_start']),
        noise_multiplier=float(env_cfg['noise_multiplier']),
    )
    agent.env = env

    bind_latest_checkpoint_paths(chkpt_dir, mod_num, agent, transformer)
    resume_from_checkpoint = bool(t.get('resume_from_checkpoint', False))
    if resume_from_checkpoint:
        print('resume_from_checkpoint=true: loading saved weights when files exist under checkpoint_dir.')
        try:
            transformer.load_checkpoint()
            transformer.to(device)
        except OSError as e:
            print('No transformer checkpoint (starting encoder/block from init):', e)
        agent.load_models(load_encoder=False)
        agent.actor.to(device)
        agent.target_actor.to(device)
        agent.critic_1.to(device)
        agent.target_critic_1.to(device)
    else:
        print('resume_from_checkpoint=false: training from randomly initialized weights.')

    grad_window = []
    loss_window = []
    loss_log_interval = int(t.get('loss_log_interval_episodes', 10))
    loss_keys = (
        'loss_kl',
        'loss_lpol',
        'loss_entropy',
        'loss_entropy_weighted',
        'loss_total',
    )
    sample_size = int(t['threshold_sample_size'])
    theta_threshold = float(t['theta_reduce_threshold'])
    theta_delta = float(t['theta_reduce_delta'])
    save_every = int(t['save_interval_episodes'])
    grad_log_every = int(t['grad_log_interval_episodes'])

    score_history = []
    correct_history = []

    state_placeholder = np.zeros(state_dim, dtype=np.float32)

    for i in range(n_games):
        state = env.reset()
        tr_memory = None

        done = False
        score = 0
        episode_correct = False

        while not done:
            t_before = int(env.t)
            frame_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            time_t = torch.tensor([t_before], device=device, dtype=torch.long)
            with torch.no_grad():
                h_mu, h_q, _, tr_memory = transformer(frame_t, tr_memory, time_idx=time_t)

            lever_action_prob = agent.choose_action(
                h_mu.detach().cpu().numpy().reshape(1, -1)
            )
            # print(lever_action_prob.shape)
            sampled_index = np.random.choice(len(lever_action_prob), p=lever_action_prob)
            lever_action_buffer = np.zeros(2)
            lever_action_buffer[sampled_index] = 1

            frame_for_state = state.copy()
            next_state, reward_env, done, _ = env.step(sampled_index)
            t_after = int(env.t)

            # Always store the true post-step observation and env clock. The episode buffer /
            # build_unroll_frames_for_td_bootstrap use next_state_frames[L-1] as s_L for the
            # extended unroll; duplicating s_{L-1} on done (old bug) made "wait until horizon"
            # on no-change trials encode the wrong frame and time at the terminal bootstrap slot.
            frame_for_next = next_state.copy()

            agent.remember(
                state_placeholder,
                lever_action_buffer,
                reward_env,
                state_placeholder,
                done,
                t_before,
                state_frames=frame_for_state,
                next_state_frames=frame_for_next,
                state_time=t_before,
                next_state_time=t_after,
            )

            learn_out = agent.learn(transformer)
            if isinstance(learn_out, tuple) and len(learn_out) == 2:
                transformer, gstats = learn_out
                if gstats is not None:
                    grad_window.append(gstats)
                    if 'loss_kl' in gstats:
                        loss_window.append({k: float(gstats[k]) for k in loss_keys if k in gstats})
            else:
                transformer = learn_out

            score += reward_env
            if reward_env > 0:
                episode_correct = True

            if not done:
                state = next_state

        score_history.append(score)
        correct_history.append(1 if episode_correct else 0)

        if (i + 1) % loss_log_interval == 0 and loss_window:
            parts = []
            for k in loss_keys:
                vals = [d[k] for d in loss_window if k in d]
                if vals:
                    parts.append(f'{k}={float(np.mean(vals)):.6f}')
            print(f"[loss] finished_episode={i + 1} n_gradient_steps={len(loss_window)} | " + ' | '.join(parts))
            loss_window.clear()

        tail = min(sample_size, len(correct_history))
        avg_score = float(np.mean(score_history[-tail:])) if tail else 0.0
        avg_correct = float(np.mean(correct_history[-tail:])) if tail else 0.0

        if (i + 1) % grad_log_every == 0 and grad_window:
            keys = [
                'total_pre_clip', 'transformer', 'critic', 'actor', 'encoder',
                'replay_buffer_alpha', 'replay_buffer_sample_beta',
            ]
            parts = []
            for k in keys:
                vals = [float(s[k]) for s in grad_window if k in s]
                if vals:
                    arr = np.array(vals)
                    parts.append(
                        f"{k}: mean={arr.mean():.4f} std={arr.std():.4f} min={arr.min():.4f} max={arr.max():.4f}"
                    )
            print(
                f"[gradients @ clip={agent.grad_clip_max_norm}] n_learn={len(grad_window)} | "
                + " | ".join(parts)
            )
            grad_window = []

        if (i + 1) % save_every == 0:
            agent.save_models()
            transformer.save_checkpoint()
            print('Theta!', env.theta)

        if avg_correct > theta_threshold and len(correct_history) > sample_size:
            env.theta -= theta_delta
            score_history = []
            correct_history = []
            print('New Theta!', env.theta)

        print(
            f"Episode {i}, Score: {score:.2f}, "
            f"Trailing-{tail} avg_correct: {avg_correct:.3f}, avg_score: {avg_score:.2f}, "
            f"Actions: {lever_action_prob}"
        )
