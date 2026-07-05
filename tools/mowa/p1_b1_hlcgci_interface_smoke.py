"""MoWA P1-b1 HLC-GCI interface smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from starVLA.dataloader.mowa import DATA_GATE
from starVLA.model.modules.mowa import MoWAHLCGCI, MoWAHLCGCIConfig


CONFIG = Path("configs/mowa/mowa_p1_b1_hlcgci_interface.yaml")
OUTPUT = Path("docs_zh/mowa/mowa_p1_b1_hlcgci_interface_smoke.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoWA P1-b1 HLC-GCI interface smoke.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_p1_b1_hlcgci_interface_smoke(args.repo_root, args.config)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def build_p1_b1_hlcgci_interface_smoke(
    repo_root: Path | str,
    config_path: Path = CONFIG,
) -> dict[str, Any]:
    root = Path(repo_root)
    cfg = OmegaConf.load(root / config_path)
    interface_cfg = cfg.interface
    model = MoWAHLCGCI(
        MoWAHLCGCIConfig(
            history_latent_dim=int(interface_cfg.history_latent_dim),
            condition_hidden_dim=int(interface_cfg.condition_hidden_dim),
            history_steps=int(interface_cfg.history_steps),
            compressed_history_dim=int(interface_cfg.compressed_history_dim),
            gate_hidden_dim=int(interface_cfg.gate_hidden_dim),
        )
    )
    batch_size = int(interface_cfg.batch_size_smoke)
    history_latent = torch.randn(
        batch_size,
        int(interface_cfg.history_steps),
        int(interface_cfg.history_latent_dim),
    )
    condition_tokens = torch.randn(
        batch_size,
        int(interface_cfg.condition_token_count),
        int(interface_cfg.condition_hidden_dim),
    )
    output = model(history_latent, condition_tokens)
    gate_min = float(output.gate_values.min().item())
    gate_max = float(output.gate_values.max().item())
    checks = {
        "config_exists": (root / config_path).is_file(),
        "compressed_history_shape_expected": tuple(output.compressed_history.shape)
        == (batch_size, int(interface_cfg.compressed_history_dim)),
        "gate_shape_expected": tuple(output.gate_values.shape)
        == (batch_size, int(interface_cfg.condition_hidden_dim)),
        "gated_condition_shape_expected": tuple(output.gated_condition_tokens.shape)
        == (
            batch_size,
            int(interface_cfg.condition_token_count),
            int(interface_cfg.condition_hidden_dim),
        ),
        "gate_values_in_unit_interval": 0.0 <= gate_min <= gate_max <= 1.0,
        "injection_policy_condition_path_only": str(interface_cfg.injection_policy) == "condition_path_only",
        "history_sampling_status_data_gate": str(cfg.status.history_sampling_status) == DATA_GATE,
        "gate_init_status_data_gate": str(cfg.status.gate_init_status) == DATA_GATE,
        "shape_status_data_gate": str(cfg.status.shape_status) == DATA_GATE,
    }
    return {
        "stage": "P1-b1",
        "task_id": "M4-001",
        "experiment_id": "E-004",
        "training_started": False,
        "checks": checks,
        "config": str(config_path),
        "observed": {
            "compressed_history_shape": tuple(output.compressed_history.shape),
            "gate_shape": tuple(output.gate_values.shape),
            "gated_condition_shape": tuple(output.gated_condition_tokens.shape),
            "gate_min": gate_min,
            "gate_max": gate_max,
        },
        "unresolved_items": [
            "This smoke validates interface shape and gate range only.",
            "History sampling policy and actual training gain remain separately gated.",
            "No StarVLA framework integration is enabled by this interface smoke.",
        ],
        "go_no_go": (
            "TBD: HLC-GCI interface smoke passed; framework integration remains gated"
            if all(checks.values())
            else "No-Go: HLC-GCI interface smoke failed"
        ),
    }


if __name__ == "__main__":
    main()
