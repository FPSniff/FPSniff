"""Entry point for training/evaluation.

Usage:
  python MainCondition.py --config configs/default.json
  python MainCondition.py --config configs/default.json --state eval --test_load_weight outputs/checkpoints/ckpt_latest.pt

All paths in the config are interpreted relative to the repository root unless absolute paths are provided.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from DiffusionFreeGuidence.TrainCondition import train, eval


def _repo_root() -> Path:
    # This file sits in the repo root.
    return Path(__file__).resolve().parent


def _load_json_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_path(v: Any, root: Path) -> Any:
    if isinstance(v, str):
        # Leave URLs and absolute paths untouched.
        p = Path(v)
        if v.startswith("http://") or v.startswith("https://") or p.is_absolute():
            return v
        return str((root / p).resolve())
    return v


def _resolve_config_paths(cfg: Dict[str, Any], root: Path) -> Dict[str, Any]:
    path_keys = {
        "img_dir",
        "train_eval_img_dir",
        "test_img_dir",
        "origin_img_dir",
        "save_dir",
        "results_dir",
        "training_load_weight",
        "test_load_weight",
    }
    out = dict(cfg)
    for k in path_keys:
        if k in out and out[k] is not None:
            out[k] = _resolve_path(out[k], root)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Posterior-Anealed Conditional Diffusion (PACD)")
    ap.add_argument("--config", type=str, default="configs/default.json", help="Path to a JSON config file")
    ap.add_argument("--state", type=str, choices=["train", "eval"], default=None, help="Override config[state]")
    ap.add_argument("--device", type=str, default=None, help="Override config[device], e.g., cuda:0 or cpu")
    ap.add_argument("--training_load_weight", type=str, default=None, help="Resume checkpoint path")
    ap.add_argument("--test_load_weight", type=str, default=None, help="Checkpoint path for evaluation")
    return ap.parse_args()


def main(model_config: Optional[Dict[str, Any]] = None) -> None:
    root = _repo_root()
    args = parse_args()

    if model_config is None:
        cfg_path = (root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
        model_config = _load_json_config(cfg_path)

    # Overrides
    if args.state is not None:
        model_config["state"] = args.state
    if args.device is not None:
        model_config["device"] = args.device
    if args.training_load_weight is not None:
        model_config["training_load_weight"] = args.training_load_weight
    if args.test_load_weight is not None:
        model_config["test_load_weight"] = args.test_load_weight

    model_config = _resolve_config_paths(model_config, root)

    if model_config["state"] == "train":
        train(model_config)
    else:
        eval(model_config)


if __name__ == "__main__":
    main()
