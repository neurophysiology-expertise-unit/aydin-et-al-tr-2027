"""
core/provenance.py — make every result file reconstructible: `save_result` embeds the full parameter dict and the
git commit hash of the code that produced it in each .npz (keys '__params__' and '__git_commit__').
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np


def git_commit(repo_root=None):
    """Full commit hash, suffixed ``-dirty`` for tracked *or untracked* changes.

    Untracked source files matter: a result must not claim a clean commit when the
    script that produced it is absent from that commit. Ignored runtime artifacts
    remain excluded by Git's normal status rules.
    """
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent
    try:
        h = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                    stderr=subprocess.DEVNULL).decode().strip()
        status = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"],
            stderr=subprocess.DEVNULL,
        )
        return h + ("-dirty" if status else "")
    except Exception:
        return "unknown"


def save_result(path, params, **arrays):
    """np.savez_compressed with the parameter dict and git hash embedded.

    `params` is stored as a JSON string under the key '__params__' and the commit under
    '__git_commit__', so `np.load(path)['__params__']` round-trips to the exact config.
    A result written any other way is, per the invariant, not to be reused.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "__params__": json.dumps(params, sort_keys=True, default=str),
        "__git_commit__": git_commit(),
    }
    np.savez_compressed(path, **arrays, **meta)
    return path


def load_result(path):
    """Return (arrays_dict, params_dict, git_commit). Raises if the provenance keys are
    missing — an unprovenanced result is a bug to surface, not to paper over."""
    z = np.load(path, allow_pickle=False)
    keys = set(z.files)
    if "__params__" not in keys or "__git_commit__" not in keys:
        raise ValueError(f"{path} has no embedded provenance; per invariant #2 it "
                         f"cannot be trusted — regenerate it with save_result().")
    params = json.loads(str(z["__params__"]))
    commit = str(z["__git_commit__"])
    arrays = {k: z[k] for k in z.files if not k.startswith("__")}
    return arrays, params, commit
