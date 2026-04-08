"""
Explicit masking for padded / post-terminal sequence positions.

- valid_mask[b, t] == 1 for real timesteps; 0 for padding after episode end.
- No gradient should flow through invalid positions: use the stop-gradient identity
    y = v * x + (1 - v) * x.detach()
  so invalid slots behave like constants in the backward graph.

- Recurrent memory is zeroed before each step t where valid[b, t] == 0 so padding
  steps never carry hidden state from real data forward into padded forwards.
"""

import torch


def expand_valid_to_match(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """valid: [B] or [B, T] -> broadcast to x shape."""
    if valid.dim() == 1:
        while valid.dim() < x.dim():
            valid = valid.unsqueeze(-1)
    elif valid.dim() == 2:
        while valid.dim() < x.dim():
            valid = valid.unsqueeze(-1)
    return valid.to(dtype=x.dtype, device=x.device)


def mask_output_stop_grad(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """
    valid: [B] or [B, T, ...] matching batch/time; 1 = keep grad, 0 = stop grad.
    """
    v = expand_valid_to_match(x, valid)
    return v * x + (1.0 - v) * x.detach()


def mask_memory_tuple(
    memory: tuple,
    valid_rows: torch.Tensor,
) -> tuple:
    """
    Zero recurrent state for batch rows where valid_rows[b] == 0 before a forward step.
    valid_rows: [B] float in {0, 1}
    """
    if memory is None:
        return None
    m = valid_rows.view(-1, 1, 1).to(dtype=memory[0].dtype, device=memory[0].device)
    return tuple(h * m for h in memory)


def unroll_transformer_frames(
    transformer,
    frames_bthwc: torch.Tensor,
    time_bt: torch.Tensor,
    valid_bt: torch.Tensor,
) -> tuple:
    """
    Step the Transformer **through time** like online execution: for each batch row (one episode),
    ``memory`` starts at ``None`` at t=0 and is updated after each frame. Uses **current** module
    weights only; hidden state is never loaded from the replay buffer.

    Inputs: RGB ``[B, T, 50, 50, 3]`` and env time indices ``[B, T]`` (padding may use 0).
    Returns ``(outputs_mu, outputs_q, attns)`` with ``outputs_*`` shape ``[B, T, patch_length * dff]``.
    ``attns`` is a length-``T`` list; each step's tensor has shape
    ``[B, 2, patch_length, num_heads, patch_length]`` (dual-path attention: mu index 0, q index 1).
    """
    B, T, _, _, _ = frames_bthwc.shape

    outs_mu = []
    outs_q = []
    attns = []
    memory = None

    for t in range(T):
        x_t = frames_bthwc[:, t, :, :, :]
        ti = time_bt[:, t].long()
        v_t = valid_bt[:, t]

        memory = mask_memory_tuple(memory, v_t)

        out_mu_t, out_q_t, A_t, memory = transformer(
            x_t,
            memory,
            time_idx=ti,
        )

        out_mu_t = mask_output_stop_grad(out_mu_t, v_t)
        out_q_t = mask_output_stop_grad(out_q_t, v_t)
        if memory is not None:
            memory = tuple(
                mask_output_stop_grad(h, v_t) for h in memory
            )
        outs_mu.append(out_mu_t)
        outs_q.append(out_q_t)
        attns.append(A_t)

    outputs_mu = torch.stack(outs_mu, dim=1)
    outputs_q = torch.stack(outs_q, dim=1)
    return outputs_mu, outputs_q, attns


def build_unroll_frames_for_td_bootstrap(
    state_frames_bthwc: torch.Tensor,
    next_state_frames_bthwc: torch.Tensor,
    time_bt: torch.Tensor,
    next_time_bt: torch.Tensor,
    lengths_b: torch.Tensor,
) -> tuple:
    """
    Build ``[B, T+1, ...]`` frame and time sequences so an unroll matches online memory and
    **shifted** encodings align with true next observations.

    For an episode with ``L`` transitions, ``state_frames`` holds ``s_0..s_{L-1}`` and
    ``next_state_frames[t]`` is ``s_{t+1}``. A plain length-``T`` unroll gives
    ``seq[t+1] = enc(s_{t+1})`` only for ``t < L-1``; at ``t = L-1`` the next observation is
    ``next_state_frames[L-1]``, not ``state_frames[L]`` (padding). This function overwrites
    index ``L`` with that frame so ``seq[L]`` encodes the real ``s_L`` for bootstrapping.

    ``valid_unroll[b, t] == 1`` for ``t <= L`` (states plus one step for the appended next frame);
    padding remains zero after that.
    """
    B, T, H, W, C = state_frames_bthwc.shape
    device = state_frames_bthwc.device
    dtype = state_frames_bthwc.dtype
    lengths_b = lengths_b.long().to(device)

    out = torch.zeros(B, T + 1, H, W, C, device=device, dtype=dtype)
    out[:, :T] = state_frames_bthwc

    b_idx = torch.arange(B, device=device)
    last_t = (lengths_b - 1).clamp(min=0)
    out[b_idx, lengths_b] = next_state_frames_bthwc[b_idx, last_t]

    time_out = torch.zeros(B, T + 1, device=device, dtype=time_bt.dtype)
    time_out[:, :T] = time_bt
    time_out[b_idx, lengths_b] = next_time_bt[b_idx, last_t]

    t_idx = torch.arange(T + 1, device=device).unsqueeze(0).expand(B, -1)
    valid_unroll = (t_idx <= lengths_b.unsqueeze(1)).to(dtype=torch.float32)

    return out, time_out, valid_unroll


def masked_mean(loss_per_element: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """loss_per_element, mask same shape; mean over valid entries only."""
    m = mask.to(dtype=loss_per_element.dtype)
    return (loss_per_element * m).sum() / (m.sum() + eps)
