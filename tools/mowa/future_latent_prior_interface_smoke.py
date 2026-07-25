"""MoWA future latent prior interface smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from starVLA.dataloader.mowa import DATA_GATE
from starVLA.model.modules.mowa import (
    MoWAFutureLatentPrior,
    MoWAFutureLatentPriorConfig,
)


CONFIG = Path("configs/mowa/mowa_future_latent_prior_interface.yaml")
OUTPUT = Path("docs_zh/mowa/mowa_future_latent_prior_interface_smoke.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoWA future latent prior interface smoke.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_future_latent_prior_interface_smoke(args.repo_root, args.config)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def build_future_latent_prior_interface_smoke(
    repo_root: Path | str,
    config_path: Path = CONFIG,
) -> dict[str, Any]:
    root = Path(repo_root)
    cfg = OmegaConf.load(root / config_path)
    interface_cfg = cfg.interface
    model = MoWAFutureLatentPrior(
        MoWAFutureLatentPriorConfig(
            current_latent_dim=int(interface_cfg.current_latent_dim),
            text_hidden_dim=int(interface_cfg.text_hidden_dim),
            hidden_dim=int(interface_cfg.hidden_dim),
            future_latent_dim=int(interface_cfg.future_latent_dim),
        )
    )
    batch_size = int(interface_cfg.batch_size_smoke)
    current_latent = torch.randn(batch_size, int(interface_cfg.current_latent_dim))
    text_hidden = torch.randn(batch_size, int(interface_cfg.text_hidden_dim))
    future_target = torch.randn(batch_size, int(interface_cfg.future_latent_dim))
    loss, losses, output = model.compute_loss(current_latent, text_hidden, future_target)
    checks = {
        "config_exists": (root / config_path).is_file(),
        "predicted_future_latent_shape_expected": tuple(output.predicted_future_latent.shape)
        == (batch_size, int(interface_cfg.future_latent_dim)),
        "future_latent_target_required": output.future_latent_target_required is True,
        "history_latent_not_used": output.history_latent_used is False,
        "loss_is_finite": bool(torch.isfinite(loss).item()),
        "input_policy_current_plus_text_only": str(interface_cfg.input_policy) == "current_latent_plus_text_only",
        "history_latent_status_data_gate": str(cfg.status.history_latent_status) == DATA_GATE,
        "latent_shape_status_data_gate": str(cfg.status.latent_shape_status) == DATA_GATE,
    }
    return {
        "stage": "future_latent_prior",
        "task_id": "M3-002",
        "experiment_id": "E-003",
        "training_started": False,
        "checks": checks,
        "config": str(config_path),
        "observed": {
            "predicted_future_latent_shape": tuple(output.predicted_future_latent.shape),
            "loss": float(loss.item()),
            "loss_keys": tuple(losses.keys()),
        },
        "unresolved_items": [
            "This smoke validates the future latent prior interface only; no latent cache builder is executed.",
            "history_latent remains blocked at future latent prior stage and must stay out of model inputs.",
            "latent shape / encoder / cache artifact statuses remain Data Gate "
            "until real Wan cache builder is approved.",
        ],
        "go_no_go": (
            "TBD: future latent prior interface smoke passed; latent cache builder remains gated"
            if all(checks.values())
            else "No-Go: future latent prior interface smoke failed"
        ),
    }


if __name__ == "__main__":
    main()
