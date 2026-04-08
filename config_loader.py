#!/usr/bin/env python3
"""Load training hyperparameters from JSON (defaults merged with user file)."""

import json
import os
from copy import deepcopy
from typing import Optional


_DEFAULTS_PATH = os.path.join(os.path.dirname(__file__), "config", "training_config.json")


def _deep_merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def load_training_config(path: Optional[str] = None) -> dict:
    """
    Load config from `path`, or from config/training_config.json next to this package.

    Notable keys: ``training.rec_vit_lr`` (full RecViT: patch CNN + transformer block),
    ``training.actor_lr`` / ``training.critic_lr`` (passed as ``alpha`` / ``beta`` to the agent),
    ``training.resume_from_checkpoint`` to control whether ``main.py`` restores saved weights.
    Legacy fallbacks: ``model.transformer_lr``, ``training.alpha`` / ``beta``.
    """
    with open(_DEFAULTS_PATH, encoding="utf-8") as f:
        base = json.load(f)
    if path is None or os.path.abspath(path) == os.path.abspath(_DEFAULTS_PATH):
        return base
    with open(path, encoding="utf-8") as f:
        override = json.load(f)
    return _deep_merge(base, override)


def token_dims_from_model(model_cfg: dict) -> tuple[int, int]:
    """Returns (token_dim, state_dim_flat) for flattened 4-patch observation."""
    e = int(model_cfg["embed_dim"])
    s = int(model_cfg["spatial_encoding_dim"])
    t = int(model_cfg["temporal_encoding_dim"])
    p = int(model_cfg["patch_length"])
    token_dim = e + s + t
    return token_dim, token_dim * p
