"""Validate MoWA E-001 launch candidate without starting training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LAUNCH_CANDIDATE_CONFIG = Path("configs/mowa/mowa_e001_starflow_ft0_launch_candidate.yaml")
COMMAND_CANDIDATE_CONFIG = Path("configs/mowa/mowa_e001_training_command_candidate.yaml")
RUNTIME_POLICY = Path("configs/mowa/mowa_e001_runtime_policy_draft.yaml")
RUNTIME_SWEEP_REPORT = Path("docs_zh/mowa/mowa_e001_full_vla_runtime_sweep_bs4_smoke.json")
MOWA_FUTURE_FEATURE_SOURCE_PATTERNS = (
    "layerwise_bridge_feature_source: mowa_future_feature_heads",
    "layerwise_bridge_feature_source: mowa_p0_fullheads",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoWA E-001 launch candidate smoke.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_e001_launch_candidate_smoke(args.repo_root)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def build_e001_launch_candidate_smoke(repo_root: Path | str) -> dict[str, Any]:
    root = Path(repo_root)
    candidate = root / LAUNCH_CANDIDATE_CONFIG
    command = root / COMMAND_CANDIDATE_CONFIG
    runtime_policy = root / RUNTIME_POLICY
    runtime_sweep = _read_json(root / RUNTIME_SWEEP_REPORT) or {}
    checks = {
        "launch_candidate_config_created": candidate.is_file(),
        "command_candidate_config_created": command.is_file(),
        "runtime_sweep_bs4_report_created": (root / RUNTIME_SWEEP_REPORT).is_file(),
        "candidate_launch_ready_false": _text_contains(candidate, "launch_ready: false"),
        "candidate_policy_confirmed_false": _text_contains(candidate, "policy_confirmed: false"),
        "command_launch_ready_false": _text_contains(command, "launch_ready: false"),
        "command_requires_human_confirmation": _text_contains(command, "requires_human_confirmation: true"),
        "command_points_to_candidate_config": _text_contains(
            command,
            "configs/mowa/mowa_e001_starflow_ft0_launch_candidate.yaml",
        ),
        "candidate_uses_starflow_vla": _text_contains(candidate, "name: StarFlowVLA"),
        "candidate_uses_layerwisefm": _text_contains(candidate, "action_model_type: LayerwiseFM"),
        "candidate_uses_future_feature_heads_source": _text_contains_any(
            candidate,
            MOWA_FUTURE_FEATURE_SOURCE_PATTERNS,
        ),
        "candidate_batch_size_4": _text_contains(candidate, "per_device_batch_size: 4"),
        "candidate_grad_accum_1": _text_contains(candidate, "gradient_accumulation_steps: 1"),
        "candidate_max_steps_1000": _text_contains(candidate, "max_train_steps: 1000"),
        "candidate_save_interval_1000": _text_contains(candidate, "save_interval: 1000"),
        "candidate_wandb_disabled": _text_contains(candidate, "disable_wandb: true"),
        "candidate_checkpoint_root_mowa_ckpt": _text_contains(candidate, "run_root_dir: playground/mowa_ckpt"),
        "runtime_policy_still_unconfirmed": _text_contains(runtime_policy, "policy_confirmed: false"),
        "runtime_sweep_bs4_passed": (
            runtime_sweep.get("bounded_runtime_sweep") is True
            and runtime_sweep.get("full_training_launch") is False
            and all((runtime_sweep.get("checks") or {}).values())
        ),
    }
    return {
        "stage": "P0",
        "experiment_id": "E-001",
        "training_started": False,
        "launch_ready": False,
        "checks": checks,
        "configs": {
            "launch_candidate": str(LAUNCH_CANDIDATE_CONFIG),
            "command_candidate": str(COMMAND_CANDIDATE_CONFIG),
            "runtime_policy": str(RUNTIME_POLICY),
            "runtime_sweep_bs4_report": str(RUNTIME_SWEEP_REPORT),
        },
        "unresolved_items": [
            "Candidate command is executable but not launch-approved.",
            "runtime_policy.policy_confirmed remains false.",
            "launch_ready remains false until human confirmation.",
            "E-006 action-gain rollout evidence remains pending.",
        ],
        "go_no_go": (
            "TBD: launch candidate is wired; training remains gated"
            if all(checks.values())
            else "No-Go: launch candidate wiring incomplete"
        ),
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _text_contains(path: Path, pattern: str) -> bool:
    if not path.is_file():
        return False
    return pattern in path.read_text(encoding="utf-8")


def _text_contains_any(path: Path, patterns: tuple[str, ...]) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    return any(pattern in text for pattern in patterns)


if __name__ == "__main__":
    main()
