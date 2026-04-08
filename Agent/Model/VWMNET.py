#!/usr/bin/env python3

"""
End-to-end learnable components: ``PatchEncoder`` (CNN), ``TransformerNetwork`` (frame → tokens →
attention + FFN + gated recurrent memory cells), actor/critic heads. See ARCHITECTURE.md in this folder.
Sequence masking for padded trajectories is handled in ``Agent/sequence_utils.py``.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

_VVC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _VVC_ROOT not in sys.path:
    sys.path.insert(0, _VVC_ROOT)
from device_utils import get_torch_device


# ---------------------------------------------------------------------------
# Patch CNN + patchify (pixels -> embed_dim per quadrant)
# ---------------------------------------------------------------------------


def patchify_frames_hwc(frames: torch.Tensor) -> torch.Tensor:
    """
    frames: [N, 50, 50, 3] -> patches [N, 4, 25, 25, 3]
    """
    n = frames.shape[0]
    p = torch.zeros((n, 4, 25, 25, 3), device=frames.device, dtype=frames.dtype)
    p[:, 0] = frames[:, 0:25, 0:25, :]
    p[:, 1] = frames[:, 25:50, 0:25, :]
    p[:, 2] = frames[:, 0:25, 25:50, :]
    p[:, 3] = frames[:, 25:50, 25:50, :]
    return p


class PatchEncoder(nn.Module):
    """
    Shared CNN over 25x25 RGB patches: three stride-2 blocks
    ``Conv -> GroupNorm -> GELU``; blocks 1–2 have no shortcut. Block 3 adds a single residual
    from the block-1 activation (two stride-2 convs, 32→64 ch) before GELU, then linear -> embed_dim.
    """

    def __init__(self, embed_dim: int = 128, name: str = "patch_encoder", chkpt_dir: str = "checkpoints"):
        super().__init__()
        self.embed_dim = embed_dim
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, name + "_td3")

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.gn1 = nn.GroupNorm(8, 32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.gn2 = nn.GroupNorm(8, 64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.gn3 = nn.GroupNorm(8, 64)
        # Residual from post-block-1 map (32×13×13) to match conv3 output (64×4×4); no norm on shortcut.
        self.skip_1_to_3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
        )

        self.fc = nn.Linear(64 * 4 * 4, embed_dim)

    def forward(self, patch_nhwc: torch.Tensor) -> torch.Tensor:
        """
        patch_nhwc: [N, 25, 25, 3] -> [N, embed_dim]
        """
        x0 = patch_nhwc.permute(0, 3, 1, 2)
        x1 = F.gelu(self.gn1(self.conv1(x0)))

        x2 = F.gelu(self.gn2(self.conv2(x1)))

        x = self.gn3(self.conv3(x2))
        x = F.gelu(x + self.skip_1_to_3(x1))

        x = x.reshape(x.shape[0], -1)
        # print(x.shape)
        x = self.fc(x)
        return x

    def save_checkpoint(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print("... saving patch encoder checkpoint ...")
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("... loading patch encoder checkpoint ...")
        self.load_state_dict(torch.load(self.checkpoint_file, map_location="cpu"))


# ---------------------------------------------------------------------------
# Recurrent transformer core and policy / value heads
# ---------------------------------------------------------------------------


class CustomLSTMCell(nn.Module):
    """
    Gated pre-update on carried ``C`` from input ``Zi``:
    ``c_pre = sigmoid(F)·C + sigmoid(I)·tanh(Z)`` with gates driven by ``C``.

    Then ``LayerNorm`` on the last dim and ``Linear(dff, dff)`` + **GELU** to produce the next carry
    (same shape as ``C``). Deterministic; no sampling or KL.
    """

    def __init__(self, patch_size, d_model, dff):
        super(CustomLSTMCell, self).__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.dff = dff
        self.input_size = self.d_model

        self.WI = nn.Linear(self.dff, self.dff)
        self.WF = nn.Linear(self.dff, self.dff)
        self.WZ = nn.Linear(self.dff, self.dff)

        self.RI = nn.Linear(self.input_size, self.dff)
        self.RF = nn.Linear(self.input_size, self.dff)
        self.RZ = nn.Linear(self.input_size, self.dff)

        self.norm_c = nn.LayerNorm(self.dff)
        self.c_proj = nn.Linear(self.dff, self.dff)

    def forward(self, Zi, Ci):
        """
        :param Zi: [batch, patch_size, d_model]
        :param Ci: [batch, patch_size, dff]
        :return: C_next [batch, patch_size, dff]
        """
        assert Zi.shape[-2:] == (self.patch_size, self.input_size)
        assert Ci.shape[-2:] == (self.patch_size, self.dff)

        C_prev = Ci
        I_tilde = self.WI(C_prev) + self.RI(Zi)
        F_tilde = self.WF(C_prev) + self.RF(Zi)
        Z_tilde = self.WZ(C_prev) + self.RZ(Zi)

        i_gate = torch.sigmoid(I_tilde)
        f_gate = torch.sigmoid(F_tilde)
        z = Z_tilde
        c_pre = f_gate * C_prev + i_gate * z

        c_norm = self.norm_c(c_pre)
        C_next = F.gelu(self.c_proj(c_norm))
        return C_next


class TransformerBlock(nn.Module):
    """
    One timestep: Pre-LN Transformer-style block (attention + position-wise FFN), then dual memory cells.

    Dual FiLM: one self-attention branch modulated by ``C_mu`` (detached) and one by ``C_q``;
    value outputs are concatenated and projected back to ``d_model`` before the residual.
    """

    def __init__(self, block_lr, input_dims=140, hidden_dim=128,
                 ffn_dim=64, n_actions=4,
                 name='transformer', chkpt_dir='checkpoints',
                 patch_length=4, input_dims_per_patch=None,
                 dff=1024, dropout=0.01):
        super(TransformerBlock, self).__init__()
        self.input_dims = input_dims_per_patch if input_dims_per_patch is not None else input_dims
        self.hidden_dim = hidden_dim
        self.d_ff = ffn_dim
        self.patch_length = patch_length
        self.n_actions = n_actions
        self.name = name
        self.chkpt_dir = chkpt_dir
        self.d_model = self.input_dims
        self.num_heads = 1
        self.d_k = self.d_model // self.num_heads
        self.dff = dff
        self.dropout = dropout

        hd = self.num_heads * self.d_k
        self.W_q = nn.Linear(self.d_model, hd)
        self.W_k = nn.Linear(self.d_model, hd)
        self.W_v = nn.Linear(self.d_model, hd)

        # Dual FiLM on Q/K/V from C_mu vs C_q: (1 + gamma) * x + beta (Perez et al.).
        self.qkv_norm_q = nn.LayerNorm(self.d_k)
        self.qkv_norm_k = nn.LayerNorm(self.d_k)
        self.qkv_norm_v = nn.LayerNorm(self.d_k)
        self.film_h_norm_mu = nn.LayerNorm(self.dff)
        self.film_h_norm_q = nn.LayerNorm(self.dff)
        fd = self.dff
        self.W_Hq_gamma_mu = nn.Linear(fd, hd)
        self.W_Hq_beta_mu = nn.Linear(fd, hd)
        self.W_Hk_gamma_mu = nn.Linear(fd, hd)
        self.W_Hk_beta_mu = nn.Linear(fd, hd)
        self.W_Hv_gamma_mu = nn.Linear(fd, hd)
        self.W_Hv_beta_mu = nn.Linear(fd, hd)
        self.W_Hq_gamma_q = nn.Linear(fd, hd)
        self.W_Hq_beta_q = nn.Linear(fd, hd)
        self.W_Hk_gamma_q = nn.Linear(fd, hd)
        self.W_Hk_beta_q = nn.Linear(fd, hd)
        self.W_Hv_gamma_q = nn.Linear(fd, hd)
        self.W_Hv_beta_q = nn.Linear(fd, hd)

        self.attn_value_fuse = nn.Linear(2 * self.d_model, self.d_model)

        # Pre-LN for self-attention and for FFN (separate norms, standard practice).
        self.norm_attn = nn.LayerNorm(self.d_model)
        self.norm_ffn = nn.LayerNorm(self.d_model)

        # Position-wise FFN: d_model -> d_ff -> d_model.
        self.ffn_w1 = nn.Linear(self.d_model, self.d_ff)
        self.ffn_w2 = nn.Linear(self.d_ff, self.d_model)

        self.dropout_attn = nn.Dropout(self.dropout * 2)
        self.ffn_dropout1 = nn.Dropout(self.dropout)
        self.ffn_dropout2 = nn.Dropout(self.dropout)

        self.LSTM_MU = CustomLSTMCell(
            patch_size=self.patch_length, d_model=self.d_model, dff=self.dff
        )

        self.LSTM_Q = CustomLSTMCell(
            patch_size=self.patch_length, d_model=self.d_model, dff=self.dff
        )

        assert ffn_dim > 0, 'ffn_dim must be positive.'

        self.optimizer = optim.Adam(
            self.parameters(),
            lr=block_lr,
            betas=(0.9, 0.99),
            eps=1e-8,
            weight_decay=1e-6
        )

    def forward(self, state_i, memory=None):
        """
        :param state_i: [batch, patch_length, input_dims] (already projected).
        :param memory: None or ``(C_mu, C_q)``, each ``[batch, patch_length, dff]``.
        :return: H_mu, H_q (carry flats), new_memory, A with A shape ``[B, 2, T, H, T]`` (mu then q).
        """
        assert self.n_actions > 0
        batch_size = state_i.shape[0]
        device = state_i.device
        assert state_i.shape == (batch_size, self.patch_length, self.input_dims)

        if memory is None:
            C_mu = torch.zeros(batch_size, self.patch_length, self.dff, device=device)
            C_q = torch.zeros(batch_size, self.patch_length, self.dff, device=device)
        else:
            C_mu, C_q = memory

        h = state_i

        # --- Pre-LN self-attention + dual FiLM (C_mu vs C_q) + fused value projection ---
        n_tokens = self.patch_length
        x_attn = self.norm_attn(h)

        q = self.W_q(x_attn).view(batch_size, n_tokens, self.num_heads, self.d_k)
        k = self.W_k(x_attn).view(batch_size, n_tokens, self.num_heads, self.d_k)
        v = self.W_v(x_attn).view(batch_size, n_tokens, self.num_heads, self.d_k)

        q = self.qkv_norm_q(q)
        k = self.qkv_norm_k(k)
        v = self.qkv_norm_v(v)

        h_mu = self.film_h_norm_mu(C_mu.detach())
        h_qc = self.film_h_norm_q(C_q)

        gq_mu = self.W_Hq_gamma_mu(h_mu).view(batch_size, n_tokens, self.num_heads, self.d_k)
        bq_mu = self.W_Hq_beta_mu(h_mu).view(batch_size, n_tokens, self.num_heads, self.d_k)
        gk_mu = self.W_Hk_gamma_mu(h_mu).view(batch_size, n_tokens, self.num_heads, self.d_k)
        bk_mu = self.W_Hk_beta_mu(h_mu).view(batch_size, n_tokens, self.num_heads, self.d_k)
        gv_mu = self.W_Hv_gamma_mu(h_mu).view(batch_size, n_tokens, self.num_heads, self.d_k)
        bv_mu = self.W_Hv_beta_mu(h_mu).view(batch_size, n_tokens, self.num_heads, self.d_k)

        q_mu = (1.0 + gq_mu) * q + bq_mu
        k_mu = (1.0 + gk_mu) * k + bk_mu
        v_mu = (1.0 + gv_mu) * v + bv_mu

        gq_q = self.W_Hq_gamma_q(h_qc).view(batch_size, n_tokens, self.num_heads, self.d_k)
        bq_q = self.W_Hq_beta_q(h_qc).view(batch_size, n_tokens, self.num_heads, self.d_k)
        gk_q = self.W_Hk_gamma_q(h_qc).view(batch_size, n_tokens, self.num_heads, self.d_k)
        bk_q = self.W_Hk_beta_q(h_qc).view(batch_size, n_tokens, self.num_heads, self.d_k)
        gv_q = self.W_Hv_gamma_q(h_qc).view(batch_size, n_tokens, self.num_heads, self.d_k)
        bv_q = self.W_Hv_beta_q(h_qc).view(batch_size, n_tokens, self.num_heads, self.d_k)

        q_q = (1.0 + gq_q) * q + bq_q
        k_q = (1.0 + gk_q) * k + bk_q
        v_q = (1.0 + gv_q) * v + bv_q

        # [B, T, H, D] throughout — no head/token transposes (avoids strided tensors before merge).
        attn_mu, A_mu = self.calculate_attention(q_mu, k_mu, v_mu)
        attn_q, A_q = self.calculate_attention(q_q, k_q, v_q)
        fused_in = torch.cat([attn_mu.flatten(2), attn_q.flatten(2)], dim=-1)
        fused = self.attn_value_fuse(fused_in)

        h = h + self.dropout_attn(fused)
        A = torch.stack([A_mu, A_q], dim=1)

        # --- Pre-LN position-wise FFN (Vaswani: Linear -> GELU -> Linear) ---
        x_ff = self.norm_ffn(h)
        ffn_mid = self.ffn_dropout1(F.gelu(self.ffn_w1(x_ff)))
        ffn_out = self.ffn_dropout2(self.ffn_w2(ffn_mid))
        h = h + ffn_out

        C_mu = self.LSTM_MU(h, C_mu)
        C_q = self.LSTM_Q(h, C_q)

        new_memory = (C_mu, C_q)
        return C_mu, C_q, new_memory, A

    def calculate_attention(self, q, k, v):
        """
        q, k, v: [batch, seq, heads, d_k]. Returns attn_out same shape, A: [batch, seq, heads, seq].
        """
        d_k_sqrt = torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32, device=q.device))
        scores = torch.einsum('bihd,bjhd->bihj', q, k) / d_k_sqrt
        A = F.softmax(scores, dim=-1)
        attn_values = torch.einsum('bihj,bjhd->bihd', A, v)
        return attn_values, A


class TransformerNetwork(nn.Module):
    """
    End-to-end per timestep: RGB frame -> PatchEncoder (CNN) -> token assembly (spatial +
    temporal one-hots) -> TransformerBlock (attention + FFN + dual recurrent memory cells).

    Call ``forward(frame_hwc, memory, time_idx)`` with ``frame_hwc`` [B, 50, 50, 3].
    Returns ``(H_mu_flat, H_q_flat, A, new_memory)`` with flats shaped ``[B, patch_length * dff]``
    and ``A`` shaped ``[B, 2, patch_length, num_heads, patch_length]`` (attention maps: mu, then q).

    ``rec_vit_lr``: single Adam learning rate for the full RecViT (``patch_encoder`` + ``transformer_block1``).
    """

    def __init__(
        self,
        rec_vit_lr: float,
        embed_dim: int,
        spatial_dim: int,
        temporal_dim: int,
        patch_encoder_name: str,
        chkpt_dir: str,
        hidden_dim: int = 128,
        ffn_dim: int = 64,
        n_actions: int = 4,
        name: str = 'transformer',
        patch_length: int = 4,
        dim_recurrent: int = 1024,
        dropout: float = 0.01,
        embedding_noise_std: float = 0.001,
    ):
        super(TransformerNetwork, self).__init__()
        self.embed_dim = embed_dim
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.token_dim = embed_dim + spatial_dim + temporal_dim
        self.input_dims = self.token_dim
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.patch_length = patch_length
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_td3')
        self.d_model = self.input_dims
        self.num_heads = 1
        self.dff = dim_recurrent
        self.dropout = dropout
        self.embedding_noise_std = embedding_noise_std
        lr = float(rec_vit_lr)

        self.patch_encoder = PatchEncoder(
            embed_dim=embed_dim, name=patch_encoder_name, chkpt_dir=chkpt_dir
        )

        self.transformer_block1 = TransformerBlock(
            lr, input_dims=self.input_dims, hidden_dim=self.hidden_dim,
            ffn_dim=self.ffn_dim,
            n_actions=self.n_actions,
            patch_length=self.patch_length,
            input_dims_per_patch=self.input_dims,
            dff=self.dff,
            dropout=self.dropout,
        )

        rec_vit_params = (
            list(self.patch_encoder.parameters())
            + list(self.transformer_block1.parameters())
        )
        self.optimizer = optim.Adam(
            rec_vit_params,
            lr=lr,
            betas=(0.9, 0.99),
            eps=1e-8,
            weight_decay=1e-6,
        )
        self.device = get_torch_device()
        self.to(self.device)

    def _tokens_from_frame(self, frame_hwc: torch.Tensor, time_idx: torch.Tensor) -> torch.Tensor:
        """
        frame_hwc: [B, 50, 50, 3]
        time_idx: [B] int64 — env timestep for temporal one-hot (clamped).
        Returns flat [B, patch_length * token_dim] before noise.
        """
        B = frame_hwc.shape[0]
        device = frame_hwc.device
        dtype = frame_hwc.dtype
        patches = patchify_frames_hwc(frame_hwc)
        emb = self.patch_encoder(patches.view(-1, 25, 25, 3)).view(B, self.patch_length, self.embed_dim)
        tokens = torch.zeros(B, self.patch_length, self.token_dim, device=device, dtype=dtype)
        tokens[:, :, :self.embed_dim] = emb
        for p in range(self.patch_length):
            tokens[:, p, self.embed_dim + p] = 1.0
        base = self.embed_dim + self.spatial_dim
        ti = time_idx.long().clamp(0, self.temporal_dim - 1)
        for b in range(B):
            tokens[b, :, base + ti[b]] = 1.0
        return tokens.view(B, -1)

    def forward(
        self,
        frame_hwc: torch.Tensor,
        memory=None,
        time_idx: torch.Tensor = None,
    ):
        """
        :param frame_hwc: [batch, 50, 50, 3] RGB (same layout as env observation).
        :param memory: None or ``(C_mu, C_q)`` from the previous step.
        :param time_idx: [batch] long — timestep index for temporal encoding. Required.
        :return: H_mu_flat, H_q_flat, A, new_memory. ``A`` is ``[B, 2, patch_length, num_heads, patch_length]``.
        """
        batch_size = frame_hwc.shape[0]
        if time_idx is None:
            time_idx = torch.zeros(batch_size, dtype=torch.long, device=frame_hwc.device)
        elif time_idx.dim() == 0:
            time_idx = time_idx.unsqueeze(0).expand(batch_size)

        flat = self._tokens_from_frame(frame_hwc, time_idx)
        state = flat.view(batch_size, self.patch_length, self.input_dims)

        H_mu, H_q, new_memory, A = self.transformer_block1(state, memory)
        return H_mu.flatten(1), H_q.flatten(1), A, new_memory

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file, map_location='cpu'))


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims=32, hidden_dim=512, fc1_dims=256, fc2_dims=128,
                 n_actions=4, name='critic', chkpt_dir='checkpoints', patch_length=4,
                 dim_recurrent=1024, num_particles=15):
        super(CriticNetwork, self).__init__()
        self.dim_recurrent = dim_recurrent
        self.input_dims = dim_recurrent
        self.hidden_dim = hidden_dim
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.patch_length = patch_length
        self.seq_length = 1
        self.n_actions = n_actions
        self.num_particles = num_particles
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_td3')

        w = self.input_dims * self.patch_length * self.seq_length
        self.fc_action_lever_projection = nn.Linear(self.n_actions, w)
        self.ln_action_lever_projection = nn.LayerNorm(w)

        self.fc1 = nn.Linear(w + w, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, fc1_dims)
        self.ln2 = nn.LayerNorm(fc1_dims)

        self.fc3 = nn.Linear(fc1_dims, fc2_dims)
        self.ln3 = nn.LayerNorm(fc2_dims)

        self.q = nn.Linear(fc2_dims, num_particles)

        self.optimizer = optim.Adam(
            self.parameters(),
            lr=beta,
            betas=(0.9, 0.99),
            eps=1e-8,
            weight_decay=1e-6
        )
        self.device = get_torch_device()
        self.to(self.device)

    def forward(self, transformer_state, action):
        if transformer_state.dim() == 3:
            transformer_state = transformer_state.flatten(0, 1)
        if action.dim() == 3:
            action = action.flatten(0, 1)
        assert transformer_state.dim() == 2 and action.dim() == 2

        action_lever = self.fc_action_lever_projection(action)
        state_action = torch.cat((transformer_state, action_lever), dim=1)

        x = F.gelu(self.ln1(self.fc1(state_action)))
        x = F.gelu(self.ln2(self.fc2(x)))
        x = F.gelu(self.ln3(self.fc3(x)))
        q = self.q(x)

        q = F.softmax(q, dim=1)
        return q

    def save_checkpoint(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file, map_location='cpu'))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims=16, hidden_dim=512, fc1_dims=256, fc2_dims=128,
                 n_actions=4, name='Actor', chkpt_dir='checkpoints', patch_length=4,
                 dim_recurrent=1024):
        super(ActorNetwork, self).__init__()
        self.dim_recurrent = dim_recurrent
        self.input_dims = dim_recurrent
        self.hidden_dim = hidden_dim
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.patch_length = patch_length
        self.seq_length = 1
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_td3')

        w = self.input_dims * self.patch_length * self.seq_length
        self.fc1 = nn.Linear(w, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, fc1_dims)
        self.ln2 = nn.LayerNorm(fc1_dims)

        self.fc3 = nn.Linear(fc1_dims, fc2_dims)
        self.ln3 = nn.LayerNorm(fc2_dims)

        self.pi_lever = nn.Linear(fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(
            self.parameters(),
            lr=alpha,
            betas=(0.9, 0.99),
            eps=1e-8,
            weight_decay=1e-6
        )
        self.device = get_torch_device()
        self.to(self.device)

    def forward(self, transformer_state):
        if transformer_state.dim() == 3:
            transformer_state = transformer_state.flatten(0, 1)
        assert transformer_state.dim() == 2

        x = F.gelu(self.ln1(self.fc1(transformer_state)))
        x = F.gelu(self.ln2(self.fc2(x)))
        x = F.gelu(self.ln3(self.fc3(x)))

        pi_lever = F.softmax(self.pi_lever(x), dim=1)
        return pi_lever

    def save_checkpoint(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file, map_location='cpu'))
