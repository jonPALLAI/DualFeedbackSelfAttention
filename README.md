# ValueValidityCue

Distributional actor–critic RL on a change-detection task (``OCDEnv``), with a **single-step recurrent Transformer** over CNN patch tokens.

## Run training

From this directory:

```bash
python3 main.py
```

Optional config path:

```bash
python3 main.py --config /path/to/training_config.json
```

Defaults load from **`config/training_config.json`**. **All** saved weights go to **`run.checkpoint_dir`** (default **`checkpoints/`** next to **`main.py`**, not the shell cwd).

On startup, **`bind_latest_checkpoint_paths`** points each module at **`checkpoint_dir/<name>_td3`** when that file exists. Actual restore is controlled by **`training.resume_from_checkpoint`**. When enabled, **`transformer.load_checkpoint()`** runs first (full **`TransformerNetwork`** including **`PatchEncoder`**), then **`agent.load_models(load_encoder=False)`** for actor/critic targets; the standalone **`patch_encoder*_td3`** is still written on save for convenience but is not loaded again after the transformer bundle.

## Layout

| Path | Role |
|------|------|
| **`main.py`** | Builds **`TransformerNetwork`**, **`Agent`**, env loop; passes RGB frames + env time into the transformer. |
| **`config/training_config.json`** | Run length, env, model dims, optimizer LRs, buffer, curriculum on **`theta`**. |
| **`config_loader.py`** | Loads and merges JSON config; **`token_dims_from_model()`** for flat token dimensions. |
| **`checkpoint_utils.py`** | **`latest_checkpoint_path`**, **`bind_latest_checkpoint_paths`** for resume. |
| **`Agent/Model/VWMNET.py`** | **PatchEncoder**, **TransformerNetwork**, **ActorNetwork**, **CriticNetwork**. |
| **`Agent/DistActCritAgent/agent.py`** | Distributional Bellman update, **episode replay** with frames, **`learn()`** unrolls the transformer on batches. |
| **`Agent/sequence_utils.py`** | **`unroll_transformer_frames`**, masking helpers for padded sequences. |
| **`Agent/episode_buffer.py`** | Padded episode batches (frames + timestep indices when using the transformer). |
| **`Agent/buffer.py`** | **`ReplayBuffer`** (prioritized experience replay); used next to **`EpisodeReplayBuffer`**. |
| **`device_utils.py`** | Torch device selection / naming helpers. |
| **`Agent/Model/patch_encoder.py`** | Re-exports **`PatchEncoder`** / **`patchify_frames_hwc`** from **`VWMNET`**. |
| **`OCDEnv.py`** | Environment. |

Dependencies are listed in **`requirements.txt`** (`pip install -r requirements.txt`).

## Model architecture (detail)

See **`Agent/Model/ARCHITECTURE.md`**.

## Config knobs (short)

- **`model.transformer_lr`**: legacy fallback LR for the full **RecViT** when **`training.rec_vit_lr`** is omitted.
- **`training.rec_vit_lr`**: Adam LR for the full **TransformerNetwork** (**PatchEncoder** + transformer block).
- **`training.resume_from_checkpoint`**: whether to restore transformer / actor / critic weights before training.
- **`training.max_seq_len`**: **`null`** or **`0`** = use **full** episode length in each padded batch (no truncation). Positive integer = cap sequence length.
- **`training.loss_log_interval_episodes`**: Every N episodes, print mean **`loss_kl`**, **`loss_lpol`**, **`loss_entropy`**, **`loss_entropy_weighted`**, **`loss_total`** over the gradient steps since the last log.
- **`model.embed_dim`**, **`spatial_encoding_dim`**, **`temporal_encoding_dim`**, **`patch_length`**: define **`token_dim`** per patch and flat state width for legacy buffer fields.

## Tests

There is no `pytest` suite checked into this repository yet (the `test_sequence_mask_grad.py` smoke test mentioned in older notes is not present). Add tests under e.g. `tests/` and document the command here when they land.
