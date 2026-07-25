"""Validate MoWA E-010 eval tracking plan without launching evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

try:
    from tools.mowa.mowa_checkpoint_resolver import resolve_mowa_checkpoint_reference
except ModuleNotFoundError:  # pragma: no cover
    from mowa_checkpoint_resolver import resolve_mowa_checkpoint_reference


CONFIG = Path("configs/mowa/mowa_e010_eval_tracking_plan.yaml")
OUTPUT = Path("docs_zh/mowa/mowa_e010_eval_tracking_smoke.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoWA E-010 eval tracking smoke.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_e010_eval_tracking_smoke(args.repo_root, args.config)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(f"go_no_go: {payload['go_no_go']}")


def build_e010_eval_tracking_smoke(
    repo_root: Path | str,
    config_path: Path = CONFIG,
) -> dict[str, Any]:
    root = Path(repo_root)
    cfg_path = root / config_path
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
    guard = cfg.get("launch_guard") or {}
    checkpoint_cfg = cfg.get("checkpoint") or {}
    constraints = cfg.get("constraints") or {}
    eval_cfg = cfg.get("eval") or {}
    checkpoint = resolve_mowa_checkpoint_reference(
        root,
        checkpoint_cfg.get("checkpoint", "latest_complete"),
        checkpoint_root_policy=checkpoint_cfg.get("checkpoint_root_policy", "playground/mowa_ckpt"),
    )
    checks = {
        "config_exists": cfg_path.is_file(),
        "launch_guard_open": (
            guard.get("launch_ready") is True
            and guard.get("policy_confirmed") is True
            and guard.get("human_confirmed") is True
        ),
        "checkpoint_exists": (root / checkpoint).is_dir(),
        "report_path_declared": bool(eval_cfg.get("report")),
        "benchmarks_declared": bool(eval_cfg.get("benchmarks")),
        "interventions_declared": bool(eval_cfg.get("interventions")),
        "counts_as_training_experiment_false": (
            constraints.get("counts_as_training_experiment") is False
        ),
        "no_checkpoint_mutation": constraints.get("no_checkpoint_mutation") is True,
        "no_training_started": constraints.get("no_training_started") is True,
    }
    return {
        "stage": "eval_tracking",
        "experiment_id": "E-010",
        "training_started": False,
        "eval_started": False,
        "launch_ready": checks["launch_guard_open"],
        "checks": checks,
        "observed": {"checkpoint": str(checkpoint)},
        "unresolved_items": [
            "This smoke validates E-010 eval tracking plan only; it does not start benchmark evaluation."
        ],
        "go_no_go": (
            "TBD: E-010 eval tracking plan is wired and launch-approved"
            if all(checks.values())
            else "No-Go: E-010 eval tracking plan incomplete"
        ),
    }


if __name__ == "__main__":
    main()
