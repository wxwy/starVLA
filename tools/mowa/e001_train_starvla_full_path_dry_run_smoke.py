"""Validate MoWA E-001 train_starvla full-path dry-run output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DRY_RUN_CONFIG = Path("configs/mowa/mowa_e001_train_starvla_full_path_dry_run.yaml")
DRY_RUN_REPORT = Path("docs_zh/mowa/mowa_e001_train_starvla_full_path_dry_run.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoWA E-001 train_starvla dry-run report smoke.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_train_starvla_full_path_dry_run_smoke(args.repo_root)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def build_train_starvla_full_path_dry_run_smoke(repo_root: Path | str) -> dict[str, Any]:
    root = Path(repo_root)
    report = _read_json(root / DRY_RUN_REPORT) or {}
    framework = report.get("framework") or {}
    forward = report.get("forward") or {}
    forward_keys = forward.get("keys") or []
    data = report.get("data") or {}
    checks = {
        "dry_run_config_created": (root / DRY_RUN_CONFIG).is_file(),
        "dry_run_report_created": (root / DRY_RUN_REPORT).is_file(),
        "entrypoint_is_train_starvla": report.get("entrypoint") == "starVLA/training/train_starvla.py",
        "full_path_dry_run_only": report.get("full_path_dry_run_only") is True,
        "training_not_started": report.get("training_started") is False,
        "checkpoint_not_saved": report.get("checkpoint_saved") is False,
        "wandb_not_started": report.get("wandb_started") is False,
        "robocasa_data_mix": data.get("data_mix") == "robocasa365_open_drawer_target_human",
        "framework_qwenoft": framework.get("name") == "QwenOFT",
        "mowa_action_bridge_probe_enabled": (
            framework.get("mowa_action_bridge_probe_enabled") is True
        ),
        "mowa_future_supervision_probe_enabled": (
            _get_future_supervision_value(framework, "probe_enabled") is True
        ),
        "mowa_future_supervision_labels_forward_evaluated": (
            _get_future_supervision_value(framework, "label_status")
            == "forward_evaluated_in_full_path_dry_run"
        ),
        "batch_fetched": (data.get("batch_summary") or {}).get("fetched") is True,
        "mowa_future_labels_enabled": data.get("mowa_future_labels_enabled") is True,
        "batch_has_mowa_future_targets": "mowa_future_targets"
        in ((data.get("batch_summary") or {}).get("first_item_keys") or []),
        "batch_has_mowa_future_masks": "mowa_future_masks"
        in ((data.get("batch_summary") or {}).get("first_item_keys") or []),
        "forward_evaluated": forward.get("evaluated") is True,
        "forward_has_mowa_future_supervision_loss": (
            "mowa_future_supervision_loss" in forward_keys
            or "mowa_p0_supervision_loss" in forward_keys
        ),
        "forward_has_action_loss": "action_loss" in forward_keys,
    }
    return {
        "stage": "full_heads",
        "experiment_id": "E-001",
        "training_started": False,
        "checks": checks,
        "reports": {
            "dry_run_config": str(DRY_RUN_CONFIG),
            "dry_run_report": str(DRY_RUN_REPORT),
        },
        "observed": {
            "run_id": report.get("run_id"),
            "framework": report.get("framework"),
            "data": report.get("data"),
            "trainer": report.get("trainer"),
        },
        "unresolved_items": [
            "This validates QwenOFT StarVLA full-path wiring, not MoWA action-bridge coupling.",
            "MoWA future supervision loss is forward-evaluated but not added to action_loss.",
            "No checkpoint/save/resume launch policy is confirmed.",
        ],
        "go_no_go": (
            "TBD: train_starvla full-path dry-run smoke passed; MoWA training integration remains gated"
            if all(checks.values())
            else "No-Go: train_starvla full-path dry-run smoke incomplete"
        ),
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _get_future_supervision_value(framework: dict[str, Any], suffix: str) -> Any:
    return framework.get(
        f"mowa_future_supervision_{suffix}",
        framework.get(f"mowa_p0_supervision_{suffix}"),
    )


def _get_future_labels_enabled(data: dict[str, Any]) -> Any:
    return data.get("mowa_future_labels_enabled", data.get("mowa_p0_labels_enabled"))


if __name__ == "__main__":
    main()
