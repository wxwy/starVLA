"""MoWA E-001 readiness smoke report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_REPORTS = {
    "recipe": Path("docs_zh/mowa/mowa_g0_robocasa365_atomic_core_recipe_smoke.json"),
    "production_preflight": Path(
        "docs_zh/mowa/g0_atomic_core_smoke/mowa_g0_atomic_core_production_preflight_smoke.json"
    ),
    "p0_train_smoke": Path("docs_zh/mowa/mowa_p0_constructible_heads_train_smoke.json"),
}
P0_FULLHEADS_INTERFACE_CONFIG = Path("configs/mowa/mowa_p0_fullheads_interface.yaml")
E001_LAUNCH_DRAFT_CONFIG = Path("configs/mowa/mowa_e001_launch_draft.yaml")
E001_RUNTIME_POLICY_DRAFT_CONFIG = Path("configs/mowa/mowa_e001_runtime_policy_draft.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoWA E-001 readiness smoke.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root. Default: current directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. When omitted, prints to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_e001_readiness_report(args.repo_root)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def build_e001_readiness_report(repo_root: Path | str) -> dict[str, Any]:
    root = Path(repo_root)
    reports = {name: _read_json(root / path) for name, path in REQUIRED_REPORTS.items()}
    missing_reports = tuple(
        name for name, payload in reports.items() if payload is None
    )

    recipe = reports.get("recipe") or {}
    preflight = reports.get("production_preflight") or {}
    p0_smoke = reports.get("p0_train_smoke") or {}

    checks = {
        "recipe_available": recipe.get("available_task_count") == 10
        and recipe.get("missing_task_count") == 0,
        "production_preflight_passed": (
            preflight.get("split_status") == "smoke_passed"
            and preflight.get("worker_status") == "smoke_passed"
            and preflight.get("distributed_sampler_status") == "smoke_passed"
            and preflight.get("future_action_leakage_status") == "smoke_passed"
            and preflight.get("failed_sample_count") == 0
        ),
        "p0_one_step_smoke_passed": (
            "train smoke passed" in str(p0_smoke.get("go_no_go", ""))
            and float(p0_smoke.get("loss_after", 1.0))
            <= float(p0_smoke.get("loss_before", 0.0))
        ),
        "p0_fullheads_interface_created": (root / P0_FULLHEADS_INTERFACE_CONFIG).is_file(),
        "e001_launch_draft_created": (root / E001_LAUNCH_DRAFT_CONFIG).is_file(),
        "e001_launch_ready_false": _text_contains(
            root / E001_LAUNCH_DRAFT_CONFIG,
            "launch_ready: false",
        ),
        "runtime_policy_draft_created": (root / E001_RUNTIME_POLICY_DRAFT_CONFIG).is_file(),
        "runtime_policy_confirmed_false": _text_contains(
            root / E001_RUNTIME_POLICY_DRAFT_CONFIG,
            "policy_confirmed: false",
        ),
    }

    unresolved_items = []
    if missing_reports:
        unresolved_items.append(f"missing_reports={','.join(missing_reports)}")
    if not _sot_docs_available(root):
        unresolved_items.append("SOT docs 00/01/02 are not available in docs_zh/mowa")
    unresolved_items.extend(
        [
            "E-001 launch draft is not executable",
            "production WAM Hz/window remain Data Gate",
            "class_mapping_status remains Data Gate",
            "runtime policy draft not confirmed",
            "resource budget remains TBD",
        ]
    )

    ready_for_launch = all(checks.values()) and not missing_reports
    return {
        "stage": "P0",
        "experiment_id": "E-001",
        "experiment_name": "P0-FullHeads vs VLA baseline",
        "training_started": False,
        "reports": {name: str(path) for name, path in REQUIRED_REPORTS.items()},
        "checks": checks,
        "observed": {
            "task_count": recipe.get("task_count"),
            "available_task_count": recipe.get("available_task_count"),
            "train_episode_count": preflight.get("train_episode_count"),
            "val_episode_count": preflight.get("val_episode_count"),
            "split_overlap_count": preflight.get("split_overlap_count"),
            "distributed_overlap_count": preflight.get("distributed_overlap_count"),
            "worker_sample_count": preflight.get("worker_sample_count"),
            "failed_sample_count": preflight.get("failed_sample_count"),
            "p0_smoke_sample_count": p0_smoke.get("sample_count"),
            "loss_before": p0_smoke.get("loss_before"),
            "loss_after": p0_smoke.get("loss_after"),
            "class_mapping_status": p0_smoke.get("class_mapping_status"),
            "p0_fullheads_interface_config": str(P0_FULLHEADS_INTERFACE_CONFIG),
            "e001_launch_draft_config": str(E001_LAUNCH_DRAFT_CONFIG),
            "e001_runtime_policy_draft_config": str(E001_RUNTIME_POLICY_DRAFT_CONFIG),
        },
        "unresolved_items": unresolved_items,
        "go_no_go": (
            "TBD: E-001 prerequisites mostly passed; launch draft exists but resource/save-resume remain Data Gate"
            if ready_for_launch
            else "No-Go: E-001 prerequisites incomplete"
        ),
        "notes": [
            "This report does not start training and does not consume E-001.",
            "Existing QwenOFT RoboCasa365 scripts are baseline/walk-through scripts, not MoWA E-001 launch config.",
            "E-001 launch requires explicit user confirmation before modifying training entry points or checkpoint logic.",
        ],
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _text_contains(path: Path, pattern: str) -> bool:
    if not path.is_file():
        return False
    return pattern in path.read_text(encoding="utf-8")


def _sot_docs_available(root: Path) -> bool:
    return all(
        (root / "docs_zh" / "mowa" / name).is_file()
        for name in (
            "00_project_proposal.md",
            "01_technical_survey.md",
            "02_detailed_design.md",
        )
    )


if __name__ == "__main__":
    main()
