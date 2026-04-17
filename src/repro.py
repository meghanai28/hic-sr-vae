import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import torch
import yaml


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _safe_git(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None
    out = result.stdout.strip()
    return out or None


def runtime_info() -> dict:
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        info["cuda_device_count"] = int(torch.cuda.device_count())
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
    git_commit = _safe_git(["git", "rev-parse", "HEAD"])
    git_branch = _safe_git(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    git_dirty = _safe_git(["git", "status", "--porcelain"])
    info["git_commit"] = git_commit
    info["git_branch"] = git_branch
    info["git_dirty"] = bool(git_dirty) if git_dirty is not None else None
    return info


def write_run_artifacts(
    out_dir: str,
    *,
    script_name: str,
    args_dict: dict,
    cfg: dict | None = None,
    extra: dict | None = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    manifest = {
        "script": script_name,
        "args": args_dict,
        "runtime": runtime_info(),
    }
    if extra:
        manifest["extra"] = extra
    manifest_path = os.path.join(out_dir, "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    if cfg is not None:
        cfg_path = os.path.join(out_dir, "resolved_config.yaml")
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
