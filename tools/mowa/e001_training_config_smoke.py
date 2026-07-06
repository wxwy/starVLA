"""MoWA E-001 training config smoke.

This validates the smoke training config and command draft without launching
training. It is intentionally text/JSON based to avoid adding YAML dependencies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TRAINING_SMOKE_CONFIG = Path("configs/mowa/mowa_e001_training_smoke.yaml")
TRAINING_COMMAND_DRAFT = Path("configs/mowa/mowa_e001_training_command_draft.yaml")
RUNTIME_POLICY = Path("configs/mowa/mowa_e001_runtime_policy_draft.yaml")
A100_THROUGHPUT_REPORT = Path("docs_zh/mowa/mowa_e001_a100_throughput_smoke.json")
FULL_VLA_RUNTIME_SWEEP_REPORT = Path("docs_zh/mowa/mowa_e001_full_vla_runtime_sweep_bs4_smoke.json")
READINESS_REPORT = Path("docs_zh/mowa/mowa_e001_readiness_smoke.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoWA E-001 training config smoke.")
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
    payload = build_e001_training_config_smoke(args.repo_root)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def build_e001_training_config_smoke(repo_root: Path | str) -> dict[str, Any]:
    root = Path(repo_root)
    throughput = _read_json(root / A100_THROUGHPUT_REPORT) or {}
    runtime_sweep = _read_json(root / FULL_VLA_RUNTIME_SWEEP_REPORT) or {}
    readiness = _read_json(root / READINESS_REPORT) or {}
    config = root / TRAINING_SMOKE_CONFIG
    command = root / TRAINING_COMMAND_DRAFT
    runtime_policy = root / RUNTIME_POLICY
    stable_candidate = throughput.get("stable_candidate") or {}

    checks = {
        "training_smoke_config_created": config.is_file(),
        "training_command_draft_created": command.is_file(),
        "runtime_policy_created": runtime_policy.is_file(),
        "a100_throughput_report_created": (root / A100_THROUGHPUT_REPORT).is_file(),
        "launch_ready_false": _text_contains(config, "launch_ready: false"),
        "training_started_false": _text_contains(config, "training_started: false"),
        "dry_run_only_true": _text_contains(config, "dry_run_only: true"),
        "checkpoint_save_disabled": _text_contains(config, "save_checkpoint_during_smoke: false"),
        "checkpoint_root_is_mowa_ckpt": (
            _text_contains(config, "run_root_dir: playground/mowa_ckpt")
            and _text_contains(runtime_policy, "run_root_dir: playground/mowa_ckpt")
            and _text_contains(runtime_policy, "local_checkpoint_root: playground/mowa_ckpt")
        ),
        "checkpoint_logic_unchanged": _text_contains(
            config,
            "checkpoint_logic_change_allowed: false",
        ),
        "command_points_to_training_smoke_config": _text_contains(
            command,
            "configs/mowa/mowa_e001_training_smoke.yaml",
        ),
        "command_still_tbd_entrypoint": _text_contains(command, "TBD_FULL_E001_ENTRYPOINT"),
        "runtime_policy_unconfirmed": _text_contains(runtime_policy, "policy_confirmed: false"),
        "checkpoint_policy_confirmed": _text_contains(runtime_policy, "checkpoint_policy_confirmed: true"),
        "logging_policy_confirmed": _text_contains(runtime_policy, "logging_policy_confirmed: true"),
        "resource_policy_unconfirmed": _text_contains(runtime_policy, "resource_policy_confirmed: false"),
        "save_resume_smoke_recorded": _text_contains(runtime_policy, "save_resume_smoke_status: passed_steps_1_to_2"),
        "eval_load_smoke_recorded": _text_contains(runtime_policy, "eval_load_smoke_status: passed_steps_2_model_load"),
        "full_vla_runtime_sweep_recorded": _text_contains(
            runtime_policy,
            "full_vla_runtime_sweep_status: passed_bs1_bs2_bs4",
        ),
        "bounded_smoke_command_not_full_launch": (
            _text_contains(command, "counted_as_full_launch: false")
            and _text_contains(command, "tools/mowa/e001_full_vla_runtime_sweep_smoke.py")
        ),
        "a100_smoke_passed": (
            throughput.get("benchmark") == "a100_throughput_smoke"
            and throughput.get("training_started") is False
            and throughput.get("checkpoint_saved") is False
            and stable_candidate.get("status") == "ok"
        ),
        "full_vla_runtime_sweep_bs4_passed": (
            runtime_sweep.get("bounded_runtime_sweep") is True
            and runtime_sweep.get("full_training_launch") is False
            and all((runtime_sweep.get("checks") or {}).values())
        ),
        "readiness_training_not_started": readiness.get("training_started") is False,
    }
    return {
        "stage": "P0",
        "experiment_id": "E-001",
        "training_started": False,
        "launch_ready": False,
        "checks": checks,
        "configs": {
            "training_smoke_config": str(TRAINING_SMOKE_CONFIG),
            "training_command_draft": str(TRAINING_COMMAND_DRAFT),
            "runtime_policy": str(RUNTIME_POLICY),
            "a100_throughput_report": str(A100_THROUGHPUT_REPORT),
            "full_vla_runtime_sweep_report": str(FULL_VLA_RUNTIME_SWEEP_REPORT),
        },
        "observed": {
            "stable_candidate": stable_candidate,
            "gpu": (throughput.get("gpu") or {}).get("name"),
            "throughput_scope": (throughput.get("model") or {}).get("module"),
        },
        "unresolved_items": [
            "full E-001 training entrypoint remains TBD",
            "runtime policy draft not confirmed",
            "checkpoint/save/resume policy not confirmed for launch",
            "MoWA checkpoint root is reserved as playground/mowa_ckpt and must not be mixed into playground/Checkpoints",
            "full VLA E-001 throughput is bounded-smoke observed only, not long-training confirmed",
        ],
        "go_no_go": (
            "TBD: training config smoke passed; full E-001 launch remains gated"
            if all(checks.values())
            else "No-Go: training config smoke prerequisites incomplete"
        ),
        "notes": [
            "This smoke validates config wiring only and does not start training.",
            "The command draft intentionally keeps the full training entrypoint as TBD.",
            "No checkpoint is saved and checkpoint/resume logic is not modified.",
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


if __name__ == "__main__":
    main()
