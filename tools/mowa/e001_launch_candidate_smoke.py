"""Validate MoWA E-001 launch candidate without starting training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from starVLA.mowa_constants import MOWA_FUTURE_FEATURE_SOURCE_ALIASES

LAUNCH_CANDIDATE_CONFIG = Path("configs/mowa/mowa_e001_starflow_ft0_launch_candidate.yaml")
COMMAND_CANDIDATE_CONFIG = Path("configs/mowa/mowa_e001_training_command_candidate.yaml")
RUNTIME_POLICY = Path("configs/mowa/mowa_e001_runtime_policy_draft.yaml")
RUNTIME_SWEEP_REPORT = Path("docs_zh/mowa/mowa_e001_full_vla_runtime_sweep_bs4_smoke.json")
MOWA_FUTURE_FEATURE_SOURCE_PATTERNS = tuple(
    f"layerwise_bridge_feature_source: {source}" for source in MOWA_FUTURE_FEATURE_SOURCE_ALIASES
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
    candidate_cfg = _load_yaml(candidate)
    command_cfg = _load_yaml(command)
    runtime_policy_cfg = _load_yaml(runtime_policy)
    runtime_sweep = _read_json(root / RUNTIME_SWEEP_REPORT) or {}
    candidate_launch_guard = _extract_launch_guard(candidate_cfg)
    command_launch_guard = _extract_launch_guard(command_cfg)
    runtime_policy_status = _extract_runtime_policy_status(runtime_policy_cfg)
    final_parameter_alignment = _build_final_parameter_alignment(
        candidate_cfg,
        command_cfg,
        runtime_policy_cfg,
    )
    checks = {
        "launch_candidate_config_created": candidate.is_file(),
        "command_candidate_config_created": command.is_file(),
        "runtime_sweep_bs4_report_created": (root / RUNTIME_SWEEP_REPORT).is_file(),
        "candidate_has_launch_guard": candidate_launch_guard["present"],
        "candidate_launch_state_consistent": _launch_guard_state_is_consistent(candidate_launch_guard),
        "command_has_launch_guard": command_launch_guard["present"],
        "command_launch_state_matches_candidate": _launch_guards_match(
            candidate_launch_guard,
            command_launch_guard,
        ),
        "command_requires_human_confirmation": (
            command_launch_guard["requires_human_confirmation"] is True
        ),
        "command_points_to_candidate_config": _text_contains(
            command,
            "configs/mowa/mowa_e001_starflow_ft0_launch_candidate.yaml",
        ),
        "candidate_uses_starflow_vla": _text_contains(candidate, "name: StarFlowVLA"),
        "candidate_uses_layerwisefm": _text_contains(candidate, "action_model_type: LayerwiseFM"),
        "candidate_enables_future_supervision_loss": _text_contains(
            candidate,
            "enable_future_supervision_loss: true",
        ),
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
        "runtime_policy_state_matches_candidate": _runtime_policy_matches_launch_candidate(
            runtime_policy_status,
            candidate_launch_guard,
        ),
        "final_parameter_alignment_passed": final_parameter_alignment["passed"],
        "runtime_sweep_bs4_passed": (
            runtime_sweep.get("bounded_runtime_sweep") is True
            and runtime_sweep.get("full_training_launch") is False
            and all((runtime_sweep.get("checks") or {}).values())
        ),
    }
    return {
        "stage": "full_heads",
        "experiment_id": "E-001",
        "training_started": False,
        "launch_ready": bool(candidate_launch_guard["launch_ready"]),
        "checks": checks,
        "configs": {
            "launch_candidate": str(LAUNCH_CANDIDATE_CONFIG),
            "command_candidate": str(COMMAND_CANDIDATE_CONFIG),
            "runtime_policy": str(RUNTIME_POLICY),
            "runtime_sweep_bs4_report": str(RUNTIME_SWEEP_REPORT),
        },
        "launch_guard": candidate_launch_guard,
        "command_launch_guard": command_launch_guard,
        "runtime_policy_status": runtime_policy_status,
        "final_parameter_alignment": final_parameter_alignment,
        "unresolved_items": _build_unresolved_items(
            candidate_launch_guard,
            runtime_policy_status,
        ),
        "go_no_go": (
            _go_no_go(candidate_launch_guard)
            if all(checks.values())
            else "No-Go: launch candidate wiring incomplete"
        ),
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> Any | None:
    if not path.is_file():
        return None
    return OmegaConf.load(path)


def _extract_launch_guard(cfg: Any | None) -> dict[str, Any]:
    return {
        "present": cfg is not None and _select(cfg, "launch_guard") is not None,
        "launch_ready": _select(cfg, "launch_guard.launch_ready"),
        "policy_confirmed": _select(cfg, "launch_guard.policy_confirmed"),
        "requires_human_confirmation": _select(cfg, "launch_guard.requires_human_confirmation"),
        "human_confirmed": _select(cfg, "launch_guard.human_confirmed"),
    }


def _extract_runtime_policy_status(cfg: Any | None) -> dict[str, Any]:
    return {
        "present": cfg is not None and _select(cfg, "status") is not None,
        "launch_ready": _select(cfg, "status.launch_ready"),
        "policy_confirmed": _select(cfg, "status.policy_confirmed"),
        "resource_policy_confirmed": _select(cfg, "status.resource_policy_confirmed"),
    }


def _launch_guard_state_is_consistent(guard: dict[str, Any]) -> bool:
    if not guard["present"]:
        return False
    launch_ready = guard["launch_ready"]
    policy_confirmed = guard["policy_confirmed"]
    requires_human_confirmation = guard["requires_human_confirmation"]
    human_confirmed = guard["human_confirmed"]
    if launch_ready is False:
        return policy_confirmed is False
    if launch_ready is True:
        if policy_confirmed is not True:
            return False
        if requires_human_confirmation is True:
            return human_confirmed is True
        return True
    return False


def _launch_guards_match(candidate_guard: dict[str, Any], command_guard: dict[str, Any]) -> bool:
    return (
        command_guard["present"]
        and candidate_guard["launch_ready"] == command_guard["launch_ready"]
        and candidate_guard["policy_confirmed"] == command_guard["policy_confirmed"]
        and candidate_guard["requires_human_confirmation"]
        == command_guard["requires_human_confirmation"]
        and candidate_guard["human_confirmed"] == command_guard["human_confirmed"]
    )


def _runtime_policy_matches_launch_candidate(
    runtime_policy_status: dict[str, Any],
    candidate_guard: dict[str, Any],
) -> bool:
    return (
        runtime_policy_status["present"]
        and runtime_policy_status["policy_confirmed"] == candidate_guard["policy_confirmed"]
        and runtime_policy_status["launch_ready"] == candidate_guard["launch_ready"]
        and (
            runtime_policy_status["resource_policy_confirmed"] is True
            if candidate_guard["launch_ready"] is True
            else True
        )
    )


def _build_unresolved_items(
    candidate_launch_guard: dict[str, Any],
    runtime_policy_status: dict[str, Any],
) -> list[str]:
    unresolved = []
    if candidate_launch_guard["launch_ready"] is True:
        unresolved.append("Candidate command is executable and launch-approved.")
    else:
        unresolved.append("Candidate command is executable but not launch-approved.")
    if runtime_policy_status["policy_confirmed"] is True:
        unresolved.append("runtime_policy.policy_confirmed is true; monitor real training and checkpoint outputs.")
    else:
        unresolved.append("runtime_policy.policy_confirmed remains false.")
    if candidate_launch_guard["launch_ready"] is True:
        unresolved.append("launch_ready is true; keep checkpoint/save-resume evidence aligned with the approved config.")
    else:
        unresolved.append("launch_ready remains false until human confirmation.")
    unresolved.append("E-006 action-gain rollout evidence remains pending.")
    return unresolved


def _go_no_go(candidate_launch_guard: dict[str, Any]) -> str:
    if candidate_launch_guard["launch_ready"] is True:
        return "TBD: launch candidate is wired and launch-approved"
    return "TBD: launch candidate is wired; training remains gated"


def _select(cfg: Any | None, dot_path: str) -> Any:
    if cfg is None:
        return None
    value = OmegaConf.select(cfg, dot_path, default=None)
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def _build_final_parameter_alignment(
    candidate: Any | None,
    command: Any | None,
    runtime_policy: Any | None,
) -> dict[str, Any]:
    candidate_batch_size = _select(candidate, "datasets.vla_data.per_device_batch_size")
    candidate_grad_accum = _select(candidate, "trainer.gradient_accumulation_steps")
    candidate_effective_batch_size = (candidate_batch_size or 0) * (candidate_grad_accum or 0)
    command_comparisons = {
        "per_device_batch_size": {
            "candidate": candidate_batch_size,
            "command_candidate": _select(command, "runtime_targets.per_device_batch_size"),
        },
        "gradient_accumulation_steps": {
            "candidate": candidate_grad_accum,
            "command_candidate": _select(command, "runtime_targets.gradient_accumulation_steps"),
        },
        "effective_batch_size": {
            "candidate": candidate_effective_batch_size,
            "command_candidate": _select(command, "runtime_targets.effective_batch_size"),
        },
        "max_train_steps": {
            "candidate": _select(candidate, "trainer.max_train_steps"),
            "command_candidate": _select(command, "runtime_targets.max_train_steps"),
        },
        "save_interval": {
            "candidate": _select(candidate, "trainer.save_interval"),
            "command_candidate": _select(command, "checkpoint_policy.save_interval"),
        },
        "run_root_dir": {
            "candidate": _select(candidate, "run_root_dir"),
            "command_candidate": _select(command, "checkpoint_policy.run_root_dir"),
        },
        "checkpoint_format": {
            "candidate": _select(candidate, "trainer.checkpoint_format"),
            "command_candidate": _select(command, "checkpoint_policy.checkpoint_format"),
        },
        "save_checkpoint_as_directory": {
            "candidate": _select(candidate, "trainer.save_checkpoint_as_directory"),
            "command_candidate": _select(command, "checkpoint_policy.save_checkpoint_as_directory"),
        },
        "disable_wandb": {
            "candidate": _select(candidate, "trainer.disable_wandb"),
            "command_candidate": _select(command, "logging.disable_wandb"),
        },
        "logging_frequency": {
            "candidate": _select(candidate, "trainer.logging_frequency"),
            "command_candidate": _select(command, "logging.logging_frequency"),
        },
    }
    runtime_traceability = {
        "per_device_batch_size": {
            "candidate": candidate_batch_size,
            "runtime_policy": _select(runtime_policy, "resource_budget.per_device_batch_size"),
            "match": _contains_expected_value(
                _select(runtime_policy, "resource_budget.per_device_batch_size"),
                candidate_batch_size,
            ),
        },
        "gradient_accumulation_steps": {
            "candidate": candidate_grad_accum,
            "runtime_policy": _select(runtime_policy, "resource_budget.gradient_accumulation_steps"),
            "match": _contains_expected_value(
                _select(runtime_policy, "resource_budget.gradient_accumulation_steps"),
                candidate_grad_accum,
            ),
        },
        "effective_batch_size": {
            "candidate": candidate_effective_batch_size,
            "runtime_policy": _select(runtime_policy, "resource_budget.effective_batch_size"),
            "match": _contains_expected_value(
                _select(runtime_policy, "resource_budget.effective_batch_size"),
                candidate_effective_batch_size,
            ),
        },
        "max_train_steps": {
            "candidate": _select(candidate, "trainer.max_train_steps"),
            "runtime_policy": _select(runtime_policy, "resource_budget.max_train_steps"),
            "match": _select(runtime_policy, "resource_budget.max_train_steps")
            in (_select(candidate, "trainer.max_train_steps"), "TBD_long_training"),
        },
        "save_interval": {
            "candidate": _select(candidate, "trainer.save_interval"),
            "runtime_policy": _select(runtime_policy, "checkpoint.save_interval"),
            "match": _contains_expected_value(
                _select(runtime_policy, "checkpoint.save_interval"),
                _select(candidate, "trainer.save_interval"),
            ),
        },
        "run_root_dir": {
            "candidate": _select(candidate, "run_root_dir"),
            "runtime_policy": _select(runtime_policy, "checkpoint.run_root_dir"),
            "match": _select(candidate, "run_root_dir")
            == _select(runtime_policy, "checkpoint.run_root_dir"),
        },
        "checkpoint_format": {
            "candidate": _select(candidate, "trainer.checkpoint_format"),
            "runtime_policy": _select(runtime_policy, "checkpoint.checkpoint_format"),
            "match": _select(candidate, "trainer.checkpoint_format")
            == _select(runtime_policy, "checkpoint.checkpoint_format"),
        },
        "save_checkpoint_as_directory": {
            "candidate": _select(candidate, "trainer.save_checkpoint_as_directory"),
            "runtime_policy": _select(runtime_policy, "checkpoint.save_checkpoint_as_directory"),
            "match": _select(candidate, "trainer.save_checkpoint_as_directory")
            == _select(runtime_policy, "checkpoint.save_checkpoint_as_directory"),
        },
        "disable_wandb": {
            "candidate": _select(candidate, "trainer.disable_wandb"),
            "runtime_policy": _select(runtime_policy, "logging.disable_wandb"),
            "match": _select(candidate, "trainer.disable_wandb")
            == _select(runtime_policy, "logging.disable_wandb"),
        },
        "logging_frequency": {
            "candidate": _select(candidate, "trainer.logging_frequency"),
            "runtime_policy": _select(runtime_policy, "logging.logging_frequency"),
            "match": _select(candidate, "trainer.logging_frequency")
            == _select(runtime_policy, "logging.logging_frequency"),
        },
    }
    command_match_results = {
        name: values["candidate"] == values["command_candidate"]
        for name, values in command_comparisons.items()
    }
    runtime_trace_results = {
        name: bool(values["match"]) for name, values in runtime_traceability.items()
    }
    return {
        "passed": all(command_match_results.values()) and all(runtime_trace_results.values()),
        "command_comparisons": command_comparisons,
        "command_matches": command_match_results,
        "runtime_policy_traceability": runtime_traceability,
        "runtime_policy_trace_matches": runtime_trace_results,
        "mismatched_fields": tuple(
            name for name, matched in command_match_results.items() if not matched
        ),
        "untraceable_runtime_policy_fields": tuple(
            name for name, matched in runtime_trace_results.items() if not matched
        ),
    }


def _contains_expected_value(observed: Any, expected: Any) -> bool:
    if observed == expected:
        return True
    if expected is None:
        return observed is None
    return str(expected) in str(observed)


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
