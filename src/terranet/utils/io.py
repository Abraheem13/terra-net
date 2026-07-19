"""Run-directory bookkeeping: config snapshot, git SHA, environment lock."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def snapshot_run(cfg: DictConfig, run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    try:
        freeze = subprocess.run(["pip", "freeze"], capture_output=True, text=True, check=True)
        (run_dir / "requirements.lock").write_text(freeze.stdout)
    except Exception:
        pass
    return run_dir


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"
