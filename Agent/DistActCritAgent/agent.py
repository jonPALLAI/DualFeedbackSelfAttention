#!/usr/bin/env python3

"""
-------------------------------------------------------------------------------
Agent class implementing a distributional reinforcement learning approach,
inspired by the paper:
"Offline Actor-Critic Reinforcement Learning Scales to Large Models."

The agent trains an Actor-Critic setup with a distributional Bellman update.
The distribution is discretized into particles, and the Critic outputs
a probability mass over these particles.

-------------------------------------------------------------------------------
"""

import os
import sys
from typing import Optional

import torch as T
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import numpy as np

_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Model'))
_AGENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _MODEL_DIR not in sys.path:
    sys.path.insert(0, _MODEL_DIR)
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)

_VVC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _VVC_ROOT not in sys.path:
    sys.path.insert(0, _VVC_ROOT)
from device_utils import get_torch_device

# Local modules (ensure these are in your Python path)
from VWMNET import ActorNetwork, CriticNetwork
from buffer import ReplayBuffer
from episode_buffer import EpisodeReplayBuffer
from sequence_utils import (
    unroll_transformer_frames,
    masked_mean,
    mask_output_stop_grad,
    build_unroll_frames_for_td_bootstrap,
)


class Agent:
    """
    The Agent class handles:
      1. Storage of transitions in a replay buffer (PER).
      2. Selection of actions via an Actor network.
      3. Critic updates via distributional RL.
      4. Target networks and soft parameter updates.
      5. Optionally integrates a Transformer state encoding.

    :param alpha: Learning rate for the actor.
    :param beta: Learning rate for the critic.
    :param input_dims: Dimension of input state (flattened obs or encoded).
    :param tau: Polyak averaging coefficient for target network updates.
    :param env: Environment placeholder or reference (not used directly here).
    :param gamma: Discount factor.
    :param warmup: Number of timesteps to use random actions before learning.
    :param n_actions: Number of discrete actions.
    :param max_size: Maximum size of replay buffer.
    :param batch_size: Mini-batch size for learning.
    :param exploration_noise: Standard deviation of noise added to actions during warmup exploration.
    """

    def __init__(
            self,
            alpha,
            beta,
            input_dims,
            tau,
            env,
            gamma=0.9,
            warmup=1000,
            n_actions=4,
            max_size=100000,
            batch_size=100,
            exploration_noise=0.01,
            name='0',
            V_max=5,
            patch_length=4,
            max_seq_len: Optional[int] = 16,
            encoder=None,
            num_particles=15,
            dim_recurrent=1024,
            actor_hidden_dim=128,
            actor_fc1_dims=64,
            actor_fc2_dims=32,
            critic_hidden_dim=256,
            critic_fc1_dims=128,
            critic_fc2_dims=64,
            eta=1.0,
            entropy_coef=0.02,
            grad_clip_max_norm=10.0,
            token_dim=140,
            embed_dim=128,
            replay_buffer_alpha=0.7,
            replay_buffer_sample_beta=0.2,
            chkpt_dir='checkpoints',
    ):
        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.noise = exploration_noise
        self.eta = eta
        self.entropy_coef = entropy_coef
        self.grad_clip_max_norm = grad_clip_max_norm
        self.token_dim = token_dim
        self.embed_dim = embed_dim
        self.replay_buffer_sample_beta = replay_buffer_sample_beta
        self.env = env

        if num_particles < 2:
            raise ValueError('num_particles must be >= 2 for distributional spacing.')

        self.encoder = encoder
        self.max_seq_len = max_seq_len
        frame_shape = (50, 50, 3) if encoder is not None else None
        # Replay buffer (Prioritized Experience Replay); one timestep per stored state (state_dim tokens)
        self.memory = ReplayBuffer(
            max_size, input_dims, n_actions, alpha=replay_buffer_alpha, frame_shape=frame_shape
        )
        self.episode_buffer = EpisodeReplayBuffer(max_episodes=max_size)
        self._ep_states = []
        self._ep_actions = []
        self._ep_rewards = []
        self._ep_next_states = []
        self._ep_dones = []
        self._ep_state_frames = []
        self._ep_next_state_frames = []
        self._ep_state_times = []
        self._ep_next_state_times = []
        # Device setup
        self.device = get_torch_device()

        # Distributional RL setup
        self.num_particles = num_particles
        self.V_max = V_max
        self.particles = [
            i * (self.V_max / (self.num_particles - 1))
            for i in range(self.num_particles)
        ]
        print("Particles:", self.particles)

        self.spacing = self.particles[1] - self.particles[0]
        print('Particle spacing:', self.spacing)

        # Actor-Critic networks (transformer_state width = patch_length * dim_recurrent, flattened)
        self.actor = ActorNetwork(
            alpha, input_dims=input_dims,
            hidden_dim=actor_hidden_dim, fc1_dims=actor_fc1_dims, fc2_dims=actor_fc2_dims,
            n_actions=n_actions,
            name='actor_planner'+name, chkpt_dir=chkpt_dir,
            patch_length=patch_length,
            dim_recurrent=dim_recurrent,
        ).to(self.device)

        self.critic_1 = CriticNetwork(
            beta, input_dims=input_dims,
            hidden_dim=critic_hidden_dim, fc1_dims=critic_fc1_dims,
            fc2_dims=critic_fc2_dims, n_actions=n_actions,
            name='critic_1_planner'+name,
            chkpt_dir=chkpt_dir,
            patch_length=patch_length,
            dim_recurrent=dim_recurrent,
            num_particles=num_particles,
        ).to(self.device)

        # Target networks
        self.target_actor = ActorNetwork(
            alpha, input_dims=input_dims,
            hidden_dim=actor_hidden_dim, fc1_dims=actor_fc1_dims, fc2_dims=actor_fc2_dims,
            n_actions=n_actions,
            name='target_actor_planner'+name,
            chkpt_dir=chkpt_dir,
            patch_length=patch_length,
            dim_recurrent=dim_recurrent,
        ).to(self.device)

        self.target_critic_1 = CriticNetwork(
            beta, input_dims=input_dims,
            hidden_dim=critic_hidden_dim, fc1_dims=critic_fc1_dims,
            fc2_dims=critic_fc2_dims, n_actions=n_actions,
            name='target_critic_1_planner'+name,
            chkpt_dir=chkpt_dir,
            patch_length=patch_length,
            dim_recurrent=dim_recurrent,
            num_particles=num_particles,
        ).to(self.device)

        # Initialize target networks to match current networks
        self.update_network_parameters(tau=1.0)

    def choose_action(self, transformer_state):
        """
        Choose an action given the current state encoding from the Transformer
        (optional) and the raw observation or additional embedding.

        :param transformer_state: Output from Transformer (or direct observation).
        :param obs: Observation or additional state data.
        :return: Numpy array of action probabilities (softmax).
        """
        if self.time_step < self.warmup:
            mu = T.tensor(np.random.normal(scale=self.noise,
                                           size=(self.n_actions,)),
                          dtype=T.float).to(self.device)
            mu = F.softmax(mu, dim=-1)
        else:
            # Actor forward pass
            transformer_state = T.as_tensor(
                transformer_state, dtype=T.float, device=self.actor.device
            )
            mu = self.actor.forward(transformer_state)[0]

        self.time_step += 1
        return mu.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done, t,
                 state_frames=None, next_state_frames=None,
                 state_time=0, next_state_time=0):
        """
        Store a transition in the replay buffer.

        :param state: Current state (single timestep, flattened placeholder if using frames).
        :param t: Stored as transition index for PER (use state_time for env timestep when possible).
        :param state_time: Env timestep index for the current frame (temporal one-hot in Transformer).
        :param next_state_time: Env timestep index after the transition (must match the frame in
            ``next_state_frames``, including terminal steps — used for temporal encoding in the
            TD unroll).
        :param state_frames: Optional (50, 50, 3) RGB frame — required for Transformer training.
        """
        if self.encoder is not None:
            self.memory.store_transition(
                state, action, reward, new_state, done, t,
                state_frames=state_frames, next_state_frames=next_state_frames,
            )
        else:
            self.memory.store_transition(
                state, action, reward, new_state, done, t
            )

        self._ep_states.append(np.asarray(state).reshape(-1))
        self._ep_actions.append(np.asarray(action))
        self._ep_rewards.append(float(reward))
        self._ep_next_states.append(np.asarray(new_state).reshape(-1))
        self._ep_dones.append(bool(done))
        if state_frames is not None:
            self._ep_state_frames.append(np.asarray(state_frames, dtype=np.float32))
            self._ep_next_state_frames.append(np.asarray(next_state_frames, dtype=np.float32))
            self._ep_state_times.append(int(state_time))
            self._ep_next_state_times.append(int(next_state_time))
        if done:
            if len(self._ep_states) > 0:
                if self.encoder is not None:
                    self.episode_buffer.append_episode(
                        self._ep_states, self._ep_actions, self._ep_rewards,
                        self._ep_next_states, self._ep_dones,
                        state_frames=self._ep_state_frames,
                        next_state_frames=self._ep_next_state_frames,
                        state_times=self._ep_state_times,
                        next_state_times=self._ep_next_state_times,
                    )
                else:
                    self.episode_buffer.append_episode(
                        self._ep_states, self._ep_actions, self._ep_rewards,
                        self._ep_next_states, self._ep_dones,
                    )
            self._ep_states = []
            self._ep_actions = []
            self._ep_rewards = []
            self._ep_next_states = []
            self._ep_dones = []
            self._ep_state_frames = []
            self._ep_next_state_frames = []
            self._ep_state_times = []
            self._ep_next_state_times = []

    @staticmethod
    def _module_grad_l2_norm(module):
        """L2 norm of flattened gradients for one module (0 if no grads)."""
        total_sq = 0.0
        for p in module.parameters():
            if p.grad is not None:
                g = p.grad.data
                total_sq += float(g.norm(2).item() ** 2)
        return total_sq ** 0.5

    def _target_critic_dists_all_actions(self, ts_flat: T.Tensor) -> T.Tensor:
        """Stack ``target_critic_1(s, one_hot_a)`` for every action; shape ``[B, n_actions, num_particles]``."""
        dists = []
        for action_idx in range(self.n_actions):
            a_onehot = T.zeros(ts_flat.size(0), self.n_actions, device=ts_flat.device, dtype=ts_flat.dtype)
            a_onehot[:, action_idx] = 1.0
            dists.append(self.target_critic_1.forward(ts_flat, a_onehot).unsqueeze(1))
        return T.cat(dists, dim=1)

    def _project_c51_bellman_target(
        self,
        q1_next: T.Tensor,
        rewards_flat: T.Tensor,
        dones_flat: T.Tensor,
        z: T.Tensor,
    ) -> T.Tensor:
        """
        C51 categorical projection onto fixed support ``z`` (same as Bellemare et al. / Rainbow).

        For **non-terminal** steps, each next atom ``z_j`` carries mass ``q1_next[:, j]`` (the
        policy-mixed next distribution). It is moved to ``Tz_j = clip(r + γ z_j, V_min, V_max)``
        and split between floor/ceil atoms with weights ``m_l, m_u`` so the **expected return**
        along that atom's mass is preserved.

        For **terminal** steps, **do not** use ``q1_next``: the Bellman target is a point mass at
        ``r`` only, projected the same way (unit weight on done rows).

        The ``l -= 1`` adjustment when ``l == u`` and ``u > 0`` avoids dropping mass when ``b`` lands
        exactly on an interior atom index.

        Rows are **renormalized** to sum to 1 (guards scatter float drift). If ``r + γ z_max`` often
        hits ``V_max``, widen ``model.V_max`` or expect bias from clipping.
        """
        delta_z = z[1] - z[0]
        v_min = z[0]
        v_max = z[-1]
        n = self.num_particles
        phat = T.zeros(q1_next.size(0), n, device=q1_next.device, dtype=q1_next.dtype)
        g = self.gamma
        not_done = (1.0 - dones_flat).to(dtype=q1_next.dtype)

        # Continuing: Rainbow-style loop (prob mass scaled by (1 - done)).
        for j in range(n):
            tz_j = rewards_flat + z[j] * not_done * g
            tz_j = tz_j.clamp(v_min, v_max)
            b = (tz_j - v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()
            l = l - ((u > 0) & (l == u)).long()
            l = l.clamp(0, n - 1)
            u = u.clamp(0, n - 1)
            same = l == u
            m_l = T.where(same, T.ones_like(b), (u.float() - b))
            m_u = T.where(same, T.zeros_like(b), (b - l.float()))
            prob_j = q1_next[:, j] * not_done
            phat.scatter_add_(1, l.unsqueeze(1), (prob_j * m_l).unsqueeze(1))
            phat.scatter_add_(1, u.unsqueeze(1), (prob_j * m_u).unsqueeze(1))

        # Terminal: scalar return r only (full unit mass on done rows).
        term = dones_flat > 0.5
        if term.any():
            tz_t = rewards_flat.clamp(v_min, v_max)
            b = (tz_t - v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()
            l = l - ((u > 0) & (l == u)).long()
            l = l.clamp(0, n - 1)
            u = u.clamp(0, n - 1)
            same = l == u
            m_l = T.where(same, T.ones_like(b), (u.float() - b))
            m_u = T.where(same, T.zeros_like(b), (b - l.float()))
            w = term.to(dtype=m_l.dtype)
            phat.scatter_add_(1, l.unsqueeze(1), (w * m_l).unsqueeze(1))
            phat.scatter_add_(1, u.unsqueeze(1), (w * m_u).unsqueeze(1))

        row_sum = phat.sum(dim=1, keepdim=True).clamp(min=1e-8)
        phat = phat / row_sum
        return phat

    def learn(self, Transformer):
        """
        Train on **whole episodes** sampled from ``episode_buffer`` (see ``sample_padded_batch``):
        each row of the batch is one full trajectory (optionally padded to a common time length).

        **Recurrent state is never read from the replay buffer.** For each sampled episode we start
        with ``memory=None`` at t=0 and unroll one step past the last stored state (true next frame
        from the transition) so ``seq[t+1]`` matches ``s_{t+1}`` for the final timestep, not padding.
        Uses the **current** Transformer parameters; nothing is backpropagated through past stored
        hidden states, only through the fresh forward graph.

        ``valid_mask`` marks real transition timesteps; padding uses stop-grad and zeroed memory carry.

        **Trajectory layout (time index ``t``):** Each episode row is an ordered list of transitions
        ``(s_t, a_t, r_t, s'_t, done_t)`` as stored by ``remember`` — so ``state_frames[:,t]`` is the
        frame at ``s_t``, ``actions[:,t]`` is ``a_t``, ``rewards[:,t]`` is the scalar reward from
        that step (your env may often be zero until the final step; the buffer still has one slot per
        transition). The Transformer unroll runs ``t = 0, 1, ...`` in order with recurrent memory;
        each timestep uses **flattened** patch recurrent states (``patch_length * dff``). Flattened
        slot ``t`` pairs ``actions_flat``, ``rewards_flat`` with **Q** / **mu** stream vectors.

        **Gradients:** Critic (``q1``, distributional KL), **LPol**, and **entropy** backprop through
        ``patch_encoder`` and the Transformer block.
        """
        if len(self.episode_buffer) < 1:
            return Transformer, None

        # max_len None = use each episode's full length (no truncation to max_seq_len).
        batch = self.episode_buffer.sample_padded_batch(
            self.batch_size, max_len=self.max_seq_len, rng=np.random.default_rng()
        )
        if batch is None:
            return Transformer, None

        B, Tm, _ = batch['states'].shape
        actions = T.tensor(batch['actions'], dtype=T.float32).to(self.device)
        rewards = T.tensor(batch['rewards'], dtype=T.float32).to(self.device)
        dones = T.tensor(batch['dones'], dtype=T.float32).to(self.device)
        valid = T.tensor(batch['valid_mask'], dtype=T.float32).to(self.device)

        if self.encoder is None or 'state_times' not in batch:
            return Transformer, None

        sf = T.tensor(batch['state_frames'], dtype=T.float32).to(self.device)
        nsf = T.tensor(batch['next_state_frames'], dtype=T.float32).to(self.device)
        st = T.tensor(batch['state_times'], dtype=T.long).to(self.device)
        nst = T.tensor(batch['next_state_times'], dtype=T.long).to(self.device)
        lengths = T.tensor(batch['lengths'], dtype=T.long).to(self.device)
        # ``lengths`` from the buffer is effective transitions per row (already ``<= Tm``); clamp defensively.
        lengths = T.minimum(lengths, T.full_like(lengths, Tm))

        sf_ext, st_ext, valid_unroll = build_unroll_frames_for_td_bootstrap(
            sf, nsf, st, nst, lengths
        )
        transformer_state_seq_mu, transformer_state_seq_q, _ = unroll_transformer_frames(
            Transformer, sf_ext, st_ext, valid_unroll
        )
        # ``valid_unroll[b,t]==1`` for unroll steps t <= L (states s_0..s_L); L = n transitions.
        # ``valid[b,tau]==1`` for transition indices tau = 0..L-1. Unroll outputs align: seq[:,tau]≈enc(s_tau),
        # seq[:,tau+1]≈enc(s_{tau+1}). Stop-grad padded *transition* slots on next-state encodings only
        # (current-slot encodings are already masked inside ``unroll_transformer_frames`` via ``valid_unroll``).
        ts_next_q = transformer_state_seq_q[:, 1 : Tm + 1, :]
        ts_next_q = mask_output_stop_grad(ts_next_q, valid)
        ts_next_mu = transformer_state_seq_mu[:, 1 : Tm + 1, :]
        ts_next_mu = mask_output_stop_grad(ts_next_mu, valid)

        ts_cur_q_flat = transformer_state_seq_q[:, :Tm, :].flatten(0, 1)
        ts_cur_mu_flat = transformer_state_seq_mu[:, :Tm, :].flatten(0, 1)
        ts_next_q_flat = ts_next_q.flatten(0, 1)
        ts_next_mu_flat = ts_next_mu.flatten(0, 1)
        actions_flat = actions.flatten(0, 1)
        rewards_flat = rewards.reshape(-1)
        dones_flat = dones.reshape(-1)
        valid_flat = valid.reshape(-1)

        q1 = self.critic_1.forward(ts_cur_q_flat, actions_flat)

        z_part = T.tensor(self.particles, device=q1.device, dtype=q1.dtype)

        with T.no_grad():
            target_actions = self.target_actor.forward(ts_next_mu_flat)
            dists_next = self._target_critic_dists_all_actions(ts_next_q_flat)
            q1_next = (target_actions.unsqueeze(-1) * dists_next).sum(dim=1)
            assert q1_next.shape == (ts_cur_q_flat.size(0), self.num_particles)

            phat = self._project_c51_bellman_target(q1_next, rewards_flat, dones_flat, z_part)

            dists_cur = self._target_critic_dists_all_actions(ts_cur_q_flat)
            Q_all = (dists_cur * z_part.view(1, 1, -1)).sum(dim=-1)
            p = self.target_actor.forward(ts_cur_mu_flat)
            KQ = T.logsumexp(
                Q_all / self.eta + T.log(p.clamp(min=1e-8)),
                dim=1,
            )

        # Cross-entropy H(phat, q) = -sum phat log q; same optimum / grad w.r.t. q as KL(phat||q),
        # and avoids phat*log(phat) NaNs when phat has exact zeros (kl_div mixes those terms).
        log_q_prob = T.log(q1.clamp(min=1e-8))

        action_probs = self.actor.forward(ts_cur_mu_flat)

        kl_elem = -(phat * log_q_prob).sum(dim=-1)
        LPol_elem = T.sum(
            p * T.exp(Q_all / self.eta - KQ.unsqueeze(1)) * T.log(action_probs + 1e-8),
            dim=1,
        )
        entropy_per = -T.sum(action_probs * T.log(action_probs + 1e-6), dim=-1)

        kl_loss = masked_mean(kl_elem, valid_flat)
        LPol = masked_mean(LPol_elem, valid_flat)
        H = masked_mean(entropy_per, valid_flat)

        Transformer.optimizer.zero_grad()
        self.critic_1.optimizer.zero_grad()
        self.actor.optimizer.zero_grad()

        critic_loss = (
            1.0 * kl_loss
            - 1.0 * LPol
            - self.entropy_coef * H
        )

        critic_loss.backward()

        ent_w = self.entropy_coef * H
        grad_stats = {
            'transformer': self._module_grad_l2_norm(Transformer),
            'critic': self._module_grad_l2_norm(self.critic_1),
            'actor': self._module_grad_l2_norm(self.actor),
            'encoder': self._module_grad_l2_norm(self.encoder) if self.encoder is not None else 0.0,
            'loss_kl': float(kl_loss.detach().item()),
            'loss_lpol': float(LPol.detach().item()),
            'loss_entropy': float(H.detach().item()),
            'loss_entropy_weighted': float(ent_w.detach().item()),
            'loss_total': float(critic_loss.detach().item()),
        }
        # Patch CNN is inside Transformer.parameters(); do not add encoder twice.
        trainable = list(Transformer.parameters()) + list(self.critic_1.parameters()) + list(self.actor.parameters())
        total_pre_clip = nn_utils.clip_grad_norm_(trainable, max_norm=self.grad_clip_max_norm)
        grad_stats['total_pre_clip'] = float(total_pre_clip.item() if hasattr(total_pre_clip, 'item') else total_pre_clip)
        grad_stats['clip_max_norm'] = self.grad_clip_max_norm

        Transformer.optimizer.step()
        self.critic_1.optimizer.step()
        self.actor.optimizer.step()

        grad_stats['replay_buffer_alpha'] = float(self.memory.alpha)
        grad_stats['replay_buffer_sample_beta'] = float(self.replay_buffer_sample_beta)

        self.learn_step_cntr += 1
        self.update_network_parameters()
        return Transformer, grad_stats

    def update_network_parameters(self, tau=None):
        """
        Soft-update (Polyak averaging) of target networks.

        :param tau: If None, use self.tau. If =1, copy weights directly.
        """
        if tau is None:
            tau = self.tau

        # Pull network parameter dictionaries
        actor_params = dict(self.actor.named_parameters())
        critic_1_params = dict(self.critic_1.named_parameters())
        target_actor_params = dict(self.target_actor.named_parameters())
        target_critic_1_params = dict(self.target_critic_1.named_parameters())

        # Update critic_1 target
        for name in critic_1_params:
            critic_1_params[name] = tau * critic_1_params[name].clone() + \
                (1 - tau) * target_critic_1_params[name].clone()

        # Update actor target
        for name in actor_params:
            actor_params[name] = tau * actor_params[name].clone() + \
                (1 - tau) * target_actor_params[name].clone()

        self.target_critic_1.load_state_dict(critic_1_params)
        self.target_actor.load_state_dict(actor_params)

    def save_models(self):
        """
        Save checkpoints for actor, target_actor, critic_1, target_critic_1.
        """
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        if self.encoder is not None:
            self.encoder.save_checkpoint()

    def load_models(self, load_encoder: bool = True):
        """
        Load checkpoints for actor, target_actor, critic_1, target_critic_1.

        Set ``load_encoder=False`` when the patch CNN was already loaded via
        ``TransformerNetwork.load_checkpoint()`` (shared weights).

        Each file is loaded independently; missing files are skipped with a short message.
        """
        modules = [
            ('actor', self.actor),
            ('target_actor', self.target_actor),
            ('critic_1', self.critic_1),
            ('target_critic_1', self.target_critic_1),
        ]
        if self.encoder is not None and load_encoder:
            modules.append(('patch_encoder', self.encoder))

        for label, module in modules:
            try:
                module.load_checkpoint()
            except OSError as e:
                print(f'[skip] {label}: no checkpoint ({e})')
