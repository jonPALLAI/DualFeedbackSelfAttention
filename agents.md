# ValueValidityCue — Agent & Environment Reference

## Environment: `ChangeDetectionEnv` (`OCDEnv.py`)

A change-detection task where an RL agent views a sequence of 50×50 RGB frames and decides whether an orientation change occurred in one of four Gabor patches.

### Observation Layout

The 50×50 canvas is divided into four 25×25 quadrants, each holding one Gabor patch:

| Patch index | Position label | Canvas region        |
|-------------|---------------|----------------------|
| 0           | S_1           | top-left `[0:25, 0:25]`     |
| 1           | S_2           | bottom-left `[25:50, 0:25]` |
| 2           | S_3           | top-right `[0:25, 25:50]`   |
| 3           | S_4           | bottom-right `[25:50, 25:50]`|

### Trial Timeline (T=7, change_time=5)

| Timestep | Frame content |
|----------|---------------|
| t=0      | Blank screen (returned by `reset()`) |
| t=1      | Colored cue (disc + proportional ring) at S_1 or S_4 |
| t=2      | Blank screen |
| t=3–4    | Four Gabor patches at their initial orientations (+ noise) |
| t=5–7    | Same Gabor array, but `orientation_change` added to `change_index` patch (if `change_true == 1`) |

### Action Space

- **Action 0**: "no change detected" — wait, trial continues
- **Action 1**: "change detected" — trial ends immediately

### Cue System

**Position**: `cue_position` is `'left'` (cue drawn at S_1) or `'right'` (cue drawn at S_4).

**Color & Reward**: The cue color determines the reward magnitude for a correct response:
- Red → 5
- Green → 3
- Blue → 1

**Proportion (ring completeness)**: Values `[1.0, 0.75, 0.5, 0.25]` control what fraction of the ring surrounds the cue disc. In the default (training) environment, proportion also governs the probability that the change occurs at the cued location vs. elsewhere.

### Change Mechanics

- `change_true`: binary flag; 1 = a change will occur, 0 = no change.
- `change_index`: which patch (0–3) receives the orientation change.
- `orientation_change`: magnitude in degrees added to the base orientation at `change_time`.
- During training, `orientation_change` is sampled from `Uniform(-theta, theta)` where `theta` starts at `theta_start` (53°) and is annealed down as performance improves.

### Forcing Conditions for Analysis

After `env.reset()`, override instance attributes before the first `env.step()`:

```python
obs = env.reset()                       # t=0, blank screen — safe to override
env.change_true = 1                     # force change present
env.change_index = 0                    # force change at S_1
env.orientation_change = delta          # specific delta in degrees
env.cue_position = 'left'              # or 'right'
env.cue_color = 'red'                  # or 'green' or 'blue'
env.proportion = 1.0                   # or 0.75, 0.5, 0.25
```

The t=0 observation is always blank, so all overrides take effect before the agent sees any meaningful stimulus.

### Reward Logic (`step()`)

- `action=1` before `change_time` → reward=0, done
- `action=1` at or after `change_time` with `change_true=1` → reward = `color_rewards[cue_color]`, done
- `action=1` at or after `change_time` with `change_true=0` → reward=0, done
- `t >= T` with `action=0` and `change_true=0` → reward = `color_rewards[cue_color]`, done (correct rejection)
- `t >= T` otherwise → reward=0, done

---

## Model Architecture

### TransformerNetwork (`VWMNET.py`)

End-to-end per-timestep processing:

1. **PatchEncoder (CNN)**: Each 25×25×3 patch → `embed_dim` (128) via three stride-2 conv blocks with GroupNorm + GELU, a skip connection from block 1 to block 3, and a final linear layer.

2. **Token Assembly**: Each patch embedding is concatenated with a one-hot spatial index (4-dim) and a one-hot temporal index (8-dim), yielding `token_dim = embed_dim + spatial_dim + temporal_dim = 140`.

3. **TransformerBlock**: Pre-LN self-attention with FiLM conditioning on Q/K/V from `concat(C_mu, C_q)` (recurrent memory carries), position-wise FFN (GELU), then two `CustomLSTMCell` updates:
   - **LSTM_MU** → produces `C_mu` (policy stream, gradients detached from attention)
   - **LSTM_Q** → produces `C_q` (value stream, full gradient flow)

4. **Outputs**: `H_mu = flatten(C_mu)` and `H_q = flatten(C_q)`, each shaped `[B, patch_length × dim_recurrent]` = `[B, 4 × 512]` = `[B, 2048]`.

### ActorNetwork

MLP with GELU: `H_mu → 128 → 64 → 32 → softmax(2)`. Outputs action probabilities `[p_wait, p_respond]`.

### CriticNetwork

Distributional (C51-style): projects action one-hot to state width, concatenates with `H_q`, passes through MLP `→ softmax(num_particles=15)`. Outputs probability mass over 15 evenly-spaced return atoms `[0, V_max/14, ..., V_max]` with `V_max=5`.

### Agent (TD3-style Distributional Actor-Critic)

- Maintains actor, critic, and their target networks (Polyak-averaged).
- Trains on whole episodes sampled from an episode replay buffer.
- Recurrent memory is never stored; each training unroll starts fresh with `memory=None`.

---

## Checkpoints

All checkpoint files live in `ValueValidityCue/checkpoints/` (configured via `run.checkpoint_dir` in `config/training_config.json`, resolved relative to `main.py`).

| Module               | File                          |
|----------------------|-------------------------------|
| TransformerNetwork   | `transformer10_td3`           |
| PatchEncoder (inside)| `patch_encoder0_td3`          |
| Actor                | `actor_planner0_td3`          |
| Target Actor         | `target_actor_planner0_td3`   |
| Critic               | `critic_1_planner0_td3`       |
| Target Critic        | `target_critic_1_planner0_td3`|

The `0` suffix comes from `mod_num=0` in the config.

### Loading for Inference

```python
transformer.load_checkpoint()       # loads full TransformerNetwork (includes PatchEncoder)
agent.load_models(load_encoder=False)  # loads actor, target_actor, critic, target_critic
                                       # skip encoder since it's already in transformer
transformer.eval()
agent.actor.eval()
```

Use `torch.no_grad()` for all forward passes during analysis. Call `agent.actor(h_mu)` directly rather than `agent.choose_action()` to avoid warmup counter side effects.

---

## Config Summary (`config/training_config.json`)

| Key | Value | Notes |
|-----|-------|-------|
| `embed_dim` | 128 | CNN output per patch |
| `spatial_encoding_dim` | 4 | one-hot for 4 patches |
| `temporal_encoding_dim` | 8 | one-hot for timesteps 0–7 |
| `patch_length` | 4 | number of spatial tokens |
| `dim_recurrent` | 512 | LSTM cell width |
| `n_actions` | 2 | wait / respond |
| `num_particles` | 15 | C51 atoms |
| `V_max` | 5 | max return atom |
| `noise_multiplier` | 5.0 | Gabor orientation noise σ |
| `theta_start` | 53 | initial max change magnitude |
