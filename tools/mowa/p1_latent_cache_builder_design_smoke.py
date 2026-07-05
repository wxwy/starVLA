"""MoWA P1 latent cache builder design smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from starVLA.dataloader.mowa import DATA_GATE, build_mowa_latent_cache_contract_smoke


CONFIG = Path("configs/mowa/mowa_p1_b0_latent_cache_builder_design.yaml")
OUTPUT = Path("docs_zh/mowa/mowa_p1_latent_cache_builder_design_smoke.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoWA P1 latent cache builder design smoke.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_p1_latent_cache_builder_design_smoke(args.repo_root, args.config)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def build_p1_latent_cache_builder_design_smoke(
    repo_root: Path | str,
    config_path: Path = CONFIG,
) -> dict[str, Any]:
    root = Path(repo_root)
    cfg = OmegaConf.load(root / config_path)
    builder_cfg = cfg.builder
    report = build_mowa_latent_cache_contract_smoke(
        Path(builder_cfg.data_root) / builder_cfg.relative_path,
        cache_root=Path(builder_cfg.cache_root),
        encoder_name=str(builder_cfg.encoder_name),
        episode_indices=tuple(int(value) for value in builder_cfg.episode_indices),
        video_keys=tuple(str(value) for value in builder_cfg.video_keys),
    ).to_dict()
    checks = {
        "config_exists": (root / config_path).is_file(),
        "contract_entries_exist": len(report["entries"]) > 0,
        "cache_keys_are_unique": report["duplicate_cache_key_count"] == 0,
        "future_action_input_forbidden": report["future_action_input_status"] == "not_used_as_input",
        "latent_shape_status_data_gate": str(cfg.status.latent_shape_status) == DATA_GATE,
        "cache_hash_status_data_gate": str(cfg.status.cache_hash_status) == DATA_GATE,
        "encoder_status_data_gate": str(cfg.status.encoder_status) == DATA_GATE,
        "latency_status_data_gate": str(cfg.status.latency_status) == DATA_GATE,
        "input_policy_future_action_forbidden": str(builder_cfg.input_policy.future_action) == "forbidden",
        "input_policy_future_rgb_target_only": str(builder_cfg.input_policy.future_rgb) == "target_only",
    }
    return {
        "stage": "P1-b0",
        "task_id": "M3-001",
        "experiment_id": "E-003",
        "training_started": False,
        "checks": checks,
        "config": str(config_path),
        "observed": {
            "dataset_path": report["dataset_path"],
            "cache_root": report["cache_root"],
            "encoder_name": report["encoder_name"],
            "entry_count": len(report["entries"]),
            "missing_video_count": report["missing_video_count"],
            "missing_cache_count": report["missing_cache_count"],
            "duplicate_cache_key_count": report["duplicate_cache_key_count"],
            "future_action_input_status": report["future_action_input_status"],
        },
        "unresolved_items": [
            "This design smoke plans cache artifacts only and does not execute Wan encoder or VAE.",
            "Latency, latent shape and cache hash remain Data Gate until real builder approval.",
            "Future RGB may exist as cache target only; it is not valid WAM input evidence.",
        ],
        "go_no_go": (
            "TBD: latent cache builder design smoke passed; real builder remains gated"
            if all(checks.values())
            else "No-Go: latent cache builder design smoke failed"
        ),
    }


if __name__ == "__main__":
    main()
