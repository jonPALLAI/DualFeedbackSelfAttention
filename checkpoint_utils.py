#!/usr/bin/env python3
"""Resolve checkpoint paths under a single training checkpoint directory."""

from __future__ import annotations

import os
from typing import Optional


def resolve_checkpoint_dir(chkpt_dir: str, script_dir: str) -> str:
    """
    If ``chkpt_dir`` is relative, resolve it under ``script_dir`` (directory containing ``main.py``)
    so saves/loads do not depend on the process current working directory.
    """
    if os.path.isabs(chkpt_dir):
        return chkpt_dir
    return os.path.join(script_dir, chkpt_dir)


def latest_checkpoint_path(chkpt_dir: str, basename: str) -> Optional[str]:
    """
    Return the path to ``chkpt_dir/basename`` if that file exists.

    All weights live in one directory (``run.checkpoint_dir`` in config, resolved next to ``main.py``).
    """
    if not chkpt_dir or not os.path.isdir(chkpt_dir):
        return None
    fp = os.path.join(chkpt_dir, basename)
    if os.path.isfile(fp):
        return fp
    return None


def bind_latest_checkpoint_paths(
    chkpt_dir: str,
    mod_num: int,
    agent,
    transformer,
) -> None:
    """
    Set ``checkpoint_file`` on each module to ``os.path.join(chkpt_dir, <basename>)`` when that
    file exists, so loads use the single checkpoint directory.
    """
    m = str(mod_num)
    mapping: list[tuple[object, str]] = [
        (transformer, f"transformer1{m}_td3"),
        (agent.actor, f"actor_planner{m}_td3"),
        (agent.target_actor, f"target_actor_planner{m}_td3"),
        (agent.critic_1, f"critic_1_planner{m}_td3"),
        (agent.target_critic_1, f"target_critic_1_planner{m}_td3"),
    ]
    if agent.encoder is not None:
        mapping.append((agent.encoder, f"patch_encoder{m}_td3"))

    for module, basename in mapping:
        path = latest_checkpoint_path(chkpt_dir, basename)
        if path is not None:
            module.checkpoint_file = path
            print(f"[checkpoint] {basename} -> {path}")
