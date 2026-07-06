"""MoWA E-001 launch draft smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


LAUNCH_DRAFT = Path("configs/mowa/mowa_e001_launch_draft.yaml")
RUNTIME_POLICY = Path("configs/mowa/mowa_e001_runtime_policy_draft.yaml")
TRAINING_COMMAND_DRAFT = Path("configs/mowa/mowa_e001_training_command_draft.yaml")
TRAINING_COMMAND_CANDIDATE = Path("configs/mowa/mowa_e001_training_command_candidate.yaml")
LAUNCH_CANDIDATE = Path("configs/mowa/mowa_e001_starflow_ft0_launch_candidate.yaml")
A100_THROUGHPUT_PLAN = Path("configs/mowa/mowa_e001_a100_throughput_smoke_plan.yaml")
READINESS_REPORT = Path("docs_zh/mowa/mowa_e001_readiness_smoke.json")
A100_THROUGHPUT_REPORT = Path("docs_zh/mowa/mowa_e001_a100_throughput_smoke.json")
LAUNCH_CANDIDATE_SMOKE_REPORT = Path("docs_zh/mowa/mowa_e001_launch_candidate_smoke.json")
STARFLOW_COMPARISON_REPORT = Path("docs_zh/mowa/mowa_e001_starflow_ft0_comparison_smoke.json")


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
    launch_draft_cfg = _read_yaml(root / LAUNCH_DRAFT) or {}
    runtime_policy_cfg = _read_yaml(root / RUNTIME_POLICY) or {}
    training_command_candidate_cfg = _read_yaml(root / TRAINING_COMMAND_CANDIDATE) or {}
    launch_candidate_cfg = _read_yaml(root / LAUNCH_CANDIDATE) or {}
    launch_draft = (root / LAUNCH_DRAFT).read_text(encoding="utf-8") if (root / LAUNCH_DRAFT).is_file() else ""
    launch_candidate_smoke = _read_json(root / LAUNCH_CANDIDATE_SMOKE_REPORT) or {}
    starflow_comparison = _read_json(root / STARFLOW_COMPARISON_REPORT) or {}
    checks = {
        "launch_draft_created": (root / LAUNCH_DRAFT).is_file(),
        "runtime_policy_created": (root / RUNTIME_POLICY).is_file(),
        "training_command_draft_created": (root / TRAINING_COMMAND_DRAFT).is_file(),
        "training_command_candidate_created": (root / TRAINING_COMMAND_CANDIDATE).is_file(),
        "launch_candidate_created": (root / LAUNCH_CANDIDATE).is_file(),
        "launch_candidate_smoke_report_created": (root / LAUNCH_CANDIDATE_SMOKE_REPORT).is_file(),
        "starflow_comparison_report_created": (root / STARFLOW_COMPARISON_REPORT).is_file(),
        "a100_throughput_plan_created": (root / A100_THROUGHPUT_PLAN).is_file(),
        "launch_draft_state_recorded": _launch_draft_state_recorded(launch_draft_cfg),
        "runtime_policy_state_consistent": _runtime_policy_state_consistent(runtime_policy_cfg),
        "checkpoint_logic_unchanged": _text_contains(
            root / RUNTIME_POLICY,
            "checkpoint_logic_change_allowed: false",
        ),
        "training_command_dry_run_only": _text_contains(root / TRAINING_COMMAND_DRAFT, "dry_run_only: true"),
        "training_command_entrypoint_tbd": _text_contains(
            root / TRAINING_COMMAND_DRAFT,
            "TBD_FULL_E001_ENTRYPOINT",
        ),
        "training_command_candidate_state_consistent": _launch_candidate_state_is_consistent(
            training_command_candidate_cfg.get("launch_guard") or {}
        ),
        "launch_candidate_state_consistent": _launch_candidate_state_is_consistent(
            launch_candidate_cfg.get("launch_guard") or {}
        ),
        "launch_candidate_smoke_passed": bool(
            (launch_candidate_smoke.get("checks") or {}).get("final_parameter_alignment_passed")
        ),
        "starflow_comparison_smoke_passed": bool(
            (starflow_comparison.get("checks") or {}).get("paired_runtime_symmetry_passed")
        ),
        "executable_training_command_candidate_recorded": _text_contains(
            root / LAUNCH_DRAFT,
            "executable training command candidate exists and is human-confirmed",
        ),
        "launch_draft_reason_current": (
            "final WAM feature source are not confirmed" not in launch_draft
            and (
                "resource policy, core SOT, and action-gain "
                "evidence are not confirmed"
                in launch_draft
                or "long-training resource policy, core SOT, "
                "and action-gain evidence remain unconfirmed for full-scale training"
                in launch_draft
            )
        ),
        "launch_blocker_mentions_action_gain_not_feature_source": (
            "action-gain evidence is not validated" in launch_draft
            and "final WAM feature source are not confirmed" not in launch_draft
        ),
        "a100_smoke_no_longer_waiting_for_a100": _text_contains(
            root / A100_THROUGHPUT_PLAN,
            "dry_run_until_on_a100: false",
        ),
        "a100_throughput_report_created": (root / A100_THROUGHPUT_REPORT).is_file(),
        "readiness_training_not_started": readiness.get("training_started") is False,
        "readiness_launch_candidate_smoke_passed": bool(
            (readiness.get("checks") or {}).get("launch_candidate_smoke_passed")
        ),
        "readiness_starflow_comparison_smoke_passed": bool(
            (readiness.get("checks") or {}).get("starflow_ft0_comparison_smoke_passed")
        ),
    }
    return {
        "stage": "P0",
        "experiment_id": "E-001",
        "training_started": False,
        "launch_ready": bool(((launch_draft_cfg.get("launch") or {}).get("launch_ready"))),
        "checks": checks,
        "drafts": {
            "launch_draft": str(LAUNCH_DRAFT),
            "runtime_policy": str(RUNTIME_POLICY),
            "training_command_draft": str(TRAINING_COMMAND_DRAFT),
            "training_command_candidate": str(TRAINING_COMMAND_CANDIDATE),
            "launch_candidate": str(LAUNCH_CANDIDATE),
            "launch_candidate_smoke_report": str(LAUNCH_CANDIDATE_SMOKE_REPORT),
            "starflow_comparison_report": str(STARFLOW_COMPARISON_REPORT),
            "a100_throughput_plan": str(A100_THROUGHPUT_PLAN),
            "a100_throughput_report": str(A100_THROUGHPUT_REPORT),
        },
        "unresolved_items": [
            "batch size 4, expected VRAM and runtime are bounded full-VLA smoke observed only, "
            "not long-training confirmed",
            "runtime policy is approved only for current bounded E-001 scope, not for broader action-gain claims",
            "executable training command candidate and launch candidate must remain synchronized with runtime policy",
            "action-gain evidence remains unvalidated",
        ],
        "go_no_go": (
            "TBD: launch draft records approved bounded E-001 state; full-scale claims remain gated"
            if all(checks.values())
            else "No-Go: launch draft prerequisites incomplete"
        ),
        "notes": [
            "This smoke does not start training.",
            "Training command draft remains dry-run-only; command candidate "
            "records the currently approved bounded launch state.",
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


def _read_yaml(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = OmegaConf.load(path)
    return OmegaConf.to_container(payload, resolve=True)


def _launch_draft_state_recorded(payload: dict[str, Any]) -> bool:
    launch = payload.get("launch") or {}
    return (
        launch.get("launch_ready") is True
        and launch.get("training_started") is True
        and launch.get("requires_human_confirmation") is True
        and launch.get("human_confirmed") is True
        and launch.get("training_completed_1000_steps") is True
    )


def _runtime_policy_state_consistent(payload: dict[str, Any]) -> bool:
    status = payload.get("status") or {}
    return (
        status.get("policy_confirmed") is True
        and status.get("launch_ready") is True
        and status.get("resource_policy_confirmed") is True
    )


def _launch_candidate_state_is_consistent(guard: dict[str, Any]) -> bool:
    if guard.get("launch_ready") is False:
        return (
            guard.get("policy_confirmed") is False
            and guard.get("requires_human_confirmation") is True
        )
    if guard.get("launch_ready") is True:
        return (
            guard.get("policy_confirmed") is True
            and guard.get("requires_human_confirmation") is True
            and guard.get("human_confirmed") is True
        )
    return False


def _text_contains(path: Path, pattern: str) -> bool:
    if not path.is_file():
        return False
    return pattern in path.read_text(encoding="utf-8")


if __name__ == "__main__":
    main()
