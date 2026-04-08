# Model stack (`VWMNET.py`)

All learnable modules for ValueValidityCue live in **`Agent/Model/VWMNET.py`**. The **`patch_encoder.py`** module is a thin re-export for convenience imports.

## End-to-end path (one environment step)

1. **Input** — RGB frame **`[B, 50, 50, 3]`** and env timestep **`time_idx`** **`[B]`** (int64), used for a temporal one-hot.

2. **`TransformerNetwork`** (single place for vision + sequence core)
   - **`patchify_frames_hwc`**: frame → four **`25×25×3`** patches (quadrants: top-left, bottom-left, top-right, bottom-right).
   - **`PatchEncoder`**: shared CNN on each patch (stride-2 conv stack + linear) → **`embed_dim`** (e.g. 128).
   - **Token assembly**: each patch is a vector of length **`token_dim = embed_dim + spatial_encoding_dim + temporal_encoding_dim`**. The CNN fills the first `embed_dim` entries; remaining entries are one-hot spatial patch index and one-hot temporal index (from `time_idx`, clamped to `temporal_encoding_dim`).
   - Optional **Gaussian noise** on tokens (scaled by `embedding_noise_std` and `hidden_dim`).
   - **`TransformerBlock`**: Pre-LN self-attention with FiLM on Q/K/V from **`concat(C_mu, C_q)`** (recurrent carries from the previous step), position-wise FFN, then **two** **`CustomLSTMCell`** memory updates (**`LSTM_MU`**, **`LSTM_Q`**). Each cell: gated pre-update **`c_pre`**, **`LayerNorm`** on **`dff`**, then **`Linear(dff→dff)`** + **GELU** for the next **`C`**. Flattened outputs are still named **`H_mu` / `H_q`** in code (they are **`C_mu` / `C_q`**), each **`patch_length * dim_recurrent`**.

3. **Policy / value**
   - **`ActorNetwork`**: MLP with **GELU** on **`H_mu`** (flattened **`C_mu`**) → action logits.
   - **`CriticNetwork`**: distributional head with **GELU** on **`H_q`** (flattened **`C_q`**) with **`num_particles`** outputs per action.

4. **Critic objective** — Distributional Bellman / C51 terms only (including C51 **`kl_div`** on particles); no separate memory KL term.

## Optimizers

- **`TransformerNetwork.optimizer`** is **Adam** on **all RecViT parameters** (**`patch_encoder`** + **`transformer_block1`**) with a single learning rate **`training.rec_vit_lr`** (default `1e-5`). If **`rec_vit_lr`** is omitted in JSON, **`main.py`** falls back to **`model.transformer_lr`**.
- **Actor** and **critic** use separate Adam optimizers with **`training.actor_lr`** and **`training.critic_lr`** (defaults `1e-4`), passed from `main.py` as agent **`alpha`** / **`beta`**; legacy keys **`training.alpha`** / **`training.beta`** still work as fallbacks.

## Checkpoints

All files live in **one** directory: **`run.checkpoint_dir`** (default **`checkpoints/`**, resolved next to `main.py`). See `config/training_config.json` → **`run.checkpoint_dir`**:


| Module            | File pattern                    |
|-------------------|---------------------------------|
| Patch CNN         | `{patch_encoder_name}_td3`      |
| Transformer (full)| `{transformer name}_td3`        |
| Actor / critic    | `*_td3` as set in each class    |

`main.py` loads/saves the **`TransformerNetwork`** checkpoint (full state dict, including **`patch_encoder`**).

## Training rollouts and memory

- **Online**: `main.py` forwards **`transformer(frame, memory, time_idx)`** each step; **`memory`** is **`(C_mu, C_q)`**. **`choose_action`** uses **`H_mu`** (flattened **`C_mu`**) only.
- **Offline / replay**: The episode buffer stores **full trajectories**. Each `learn` samples a **batch of whole episodes** (rows), time running **start → end**. **`unroll_transformer_frames`** runs the Transformer with **`memory=None`** at the first real timestep of each row and updates memory **only through the forward pass** using **current** weights. No recurrent tensors are stored in the buffer or replayed from disk.
- **TD bootstrap for `s'`**: The value target uses **`transformer_state_seq[:, 1:]`** (padded with a zero column), i.e. the encoding at **`t+1`** from that same unroll, so **`s'`** matches the **same** trajectory memory as online.
- **`training.max_seq_len`**: Set to **`null`** or **`0`** in JSON to **not** truncate episodes when building padded batches (time length = longest episode in the sampled batch). A positive integer caps the prefix length.
