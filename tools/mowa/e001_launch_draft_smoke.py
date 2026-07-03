"""MoWA E-001 launch draft smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LAUNCH_DRAFT = Path("configs/mowa/mowa_e001_launch_draft.yaml")
RUNTIME_POLICY = Path("configs/mowa/mowa_e001_runtime_policy_draft.yaml")
TRAINING_COMMAND_DRAFT = Path("configs/mowa/mowa_e001_training_command_draft.yaml")
A100_THROUGHPUT_PLAN = Path("configs/mowa/mowa_e001_a100_throughput_smoke_plan.yaml")
READINESS_REPORT = Path("docs_zh/mowa/mowa_e001_readiness_smoke.json")
A100_THROUGHPUT_REPORT = Path("docs_zh/mowa/mowa_e001_a100_throughput_smoke.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoWA E-001 launch draft smoke.")
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


def build_e001_launch_draft_smoke(repo_root: Path | str) -> dict[str, Any]:
    root = Path(repo_root)
    readiness = _read_json(root / READINESS_REPORT) or {}
    checks = {
        "launch_draft_created": (root / LAUNCH_DRAFT).is_file(),
        "runtime_policy_created": (root / RUNTIME_POLICY).is_file(),
        "training_command_draft_created": (root / TRAINING_COMMAND_DRAFT).is_file(),
        "a100_throughput_plan_created": (root / A100_THROUGHPUT_PLAN).is_file(),
        "launch_ready_false": _text_contains(root / LAUNCH_DRAFT, "launch_ready: false"),
        "training_started_false": _text_contains(root / LAUNCH_DRAFT, "training_started: false"),
        "runtime_policy_unconfirmed": _text_contains(root / RUNTIME_POLICY, "policy_confirmed: false"),
        "checkpoint_logic_unchanged": _text_contains(
            root / RUNTIME_POLICY,
            "checkpoint_logic_change_allowed: false",
        ),
        "training_command_dry_run_only": _text_contains(root / TRAINING_COMMAND_DRAFT, "dry_run_only: true"),
        "a100_smoke_no_longer_waiting_for_a100": _text_contains(
            root / A100_THROUGHPUT_PLAN,
            "dry_run_until_on_a100: false",
        ),
        "a100_throughput_report_created": (root / A100_THROUGHPUT_REPORT).is_file(),
        "readiness_training_not_started": readiness.get("training_started") is False,
    }
    return {
        "stage": "P0",
        "experiment_id": "E-001",
        "training_started": False,
        "launch_ready": False,
        "checks": checks,
        "drafts": {
            "launch_draft": str(LAUNCH_DRAFT),
            "runtime_policy": str(RUNTIME_POLICY),
            "training_command_draft": str(TRAINING_COMMAND_DRAFT),
            "a100_throughput_plan": str(A100_THROUGHPUT_PLAN),
            "a100_throughput_report": str(A100_THROUGHPUT_REPORT),
        },
        "unresolved_items": [
            "class_mapping_status remains Data Gate",
            "batch size, expected VRAM and runtime are smoke-observed only, not production-confirmed",
            "runtime policy remains unconfirmed",
            "real MoWA E-001 training config is not executable",
        ],
        "go_no_go": (
            "TBD: launch drafts available; A100 smoke passed but runtime policy remains Data Gate"
            if all(checks.values())
            else "No-Go: launch draft prerequisites incomplete"
        ),
        "notes": [
            "This smoke does not start training.",
            "Training command draft is dry-run-only and keeps launch_ready=false.",
        ],
    }


def main() -> None:
    args = parse_args()
    payload = build_e001_launch_draft_smoke(args.repo_root)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _text_contains(path: Path, pattern: str) -> bool:
    if not path.is_file():
        return False
    return pattern in path.read_text(encoding="utf-8")


if __name__ == "__main__":
    main()
