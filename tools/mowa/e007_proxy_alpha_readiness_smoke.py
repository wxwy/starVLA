"""Validate MoWA E-007 proxy-alpha training candidate without starting training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


CONFIG = Path("configs/mowa/mowa_e007_proxy_alpha_candidate.yaml")
OUTPUT = Path("docs_zh/mowa/mowa_e007_proxy_alpha_readiness_smoke.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoWA E-007 proxy-alpha readiness smoke.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_e007_proxy_alpha_readiness_smoke(args.repo_root, args.config)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(f"go_no_go: {payload['go_no_go']}")


def build_e007_proxy_alpha_readiness_smoke(
    repo_root: Path | str,
    config_path: Path = CONFIG,
) -> dict[str, Any]:
    root = Path(repo_root)
    cfg_path = root / config_path
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
    guard = cfg.get("launch_guard") or {}
    inputs = cfg.get("inputs") or {}
    output = cfg.get("output") or {}
    constraints = cfg.get("constraints") or {}
    label_distribution = root / str(inputs.get("label_distribution_report", ""))
    label_manifest = root / str(inputs.get("label_cache_manifest", ""))
    checks = {
        "config_exists": cfg_path.is_file(),
        "launch_guard_open": (
            guard.get("launch_ready") is True
            and guard.get("policy_confirmed") is True
            and guard.get("human_confirmed") is True
        ),
        "label_distribution_report_exists": label_distribution.is_file(),
        "label_cache_manifest_exists": label_manifest.is_file(),
        "alpha_report_path_declared": bool(output.get("alpha_report")),
        "frozen_alpha_config_path_declared": bool(output.get("frozen_alpha_config")),
        "no_p0_p1_model_weight_update": constraints.get("no_p0_p1_model_weight_update") is True,
        "alpha_frozen_before_downstream_training": (
            constraints.get("alpha_frozen_before_downstream_training") is True
        ),
        "uniform_alpha_baseline_required": constraints.get("uniform_alpha_baseline_required") is True,
    }
    return {
        "stage": "data_mix",
        "experiment_id": "E-007",
        "training_started": False,
        "launch_ready": checks["launch_guard_open"],
        "checks": checks,
        "configs": {"proxy_alpha_candidate": str(config_path)},
        "unresolved_items": [
            "This smoke validates the E-007 proxy-alpha candidate; it does not run proxy optimization.",
            "Downstream P0/P1 configs may consume alpha only after the alpha report is reviewed and frozen.",
        ],
        "go_no_go": (
            "TBD: E-007 proxy-alpha candidate is wired and launch-approved"
            if all(checks.values())
            else "No-Go: E-007 proxy-alpha candidate incomplete"
        ),
    }


if __name__ == "__main__":
    main()
