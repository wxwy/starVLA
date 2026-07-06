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
    "production_window_preflight": Path(
        "docs_zh/mowa/g0_atomic_core_smoke/mowa_g0_atomic_core_production_window_5hz_preflight_smoke.json"
    ),
    "full_heads_train_smoke": Path("docs_zh/mowa/mowa_full_heads_constructible_train_smoke.json"),
}
FULL_HEADS_INTERFACE_CONFIG = Path("configs/mowa/mowa_full_heads_interface.yaml")
E001_LAUNCH_DRAFT_CONFIG = Path("configs/mowa/mowa_e001_launch_draft.yaml")
E001_RUNTIME_POLICY_DRAFT_CONFIG = Path("configs/mowa/mowa_e001_runtime_policy_draft.yaml")
MOWA_ACTION_BRIDGE_INTERFACE_CONFIG = Path("configs/mowa/mowa_action_bridge_interface.yaml")
E001_TRAINING_COMMAND_DRAFT_CONFIG = Path("configs/mowa/mowa_e001_training_command_draft.yaml")
E001_TRAINING_COMMAND_CANDIDATE_CONFIG = Path("configs/mowa/mowa_e001_training_command_candidate.yaml")
E001_LAUNCH_CANDIDATE_CONFIG = Path("configs/mowa/mowa_e001_starflow_ft0_launch_candidate.yaml")
E001_LAUNCH_CANDIDATE_SMOKE_REPORT = Path("docs_zh/mowa/mowa_e001_launch_candidate_smoke.json")
E001_STARFLOW_FT0_BASELINE_CANDIDATE_CONFIG = Path(
    "configs/mowa/mowa_e001_starflow_ft0_baseline_candidate.yaml"
)
E001_STARFLOW_FT0_COMPARISON_SMOKE_REPORT = Path(
    "docs_zh/mowa/mowa_e001_starflow_ft0_comparison_smoke.json"
)
E001_TRAINING_SMOKE_CONFIG = Path("configs/mowa/mowa_e001_training_smoke.yaml")
E001_TRAINING_CONFIG_SMOKE_REPORT = Path("docs_zh/mowa/mowa_e001_training_config_smoke.json")
E001_TRAIN_STARVLA_DRY_RUN_CONFIG = Path("configs/mowa/mowa_e001_train_starvla_full_path_dry_run.yaml")
E001_TRAIN_STARVLA_DRY_RUN_SMOKE_REPORT = Path(
    "docs_zh/mowa/mowa_e001_train_starvla_full_path_dry_run_smoke.json"
)
E001_STARFLOW_FT0_DRY_RUN_CONFIG = Path("configs/mowa/mowa_e001_starflow_ft0_full_path_dry_run.yaml")
E001_STARFLOW_FT0_DRY_RUN_SMOKE_REPORT = Path(
    "docs_zh/mowa/mowa_e001_starflow_ft0_full_path_dry_run_smoke.json"
)
E001_STARFLOW_FT0_TRAINING_THROUGHPUT_CONFIG = Path(
    "configs/mowa/mowa_e001_starflow_ft0_training_throughput_smoke.yaml"
)
E001_STARFLOW_FT0_TRAINING_THROUGHPUT_SMOKE_REPORT = Path(
    "docs_zh/mowa/mowa_e001_starflow_ft0_training_throughput_smoke_check.json"
)
E001_A100_THROUGHPUT_SMOKE_PLAN_CONFIG = Path("configs/mowa/mowa_e001_a100_throughput_smoke_plan.yaml")
E001_A100_THROUGHPUT_SMOKE_REPORT = Path("docs_zh/mowa/mowa_e001_a100_throughput_smoke.json")
E001_FULL_VLA_RUNTIME_SWEEP_REPORT = Path(
    "docs_zh/mowa/mowa_e001_full_vla_runtime_sweep_bs4_smoke.json"
)
E006_COUPLING_INTERVENTION_SMOKE_REPORT = Path(
    "docs_zh/mowa/mowa_e006_coupling_intervention_smoke.json"
)
E006_EVAL_LOAD_SMOKE_CONFIG = Path("configs/mowa/mowa_e006_eval_load_smoke.yaml")
E006_EVAL_LOAD_SMOKE_REPORT = Path("docs_zh/mowa/mowa_e006_eval_load_smoke.json")
E006_POLICY_ROLLOUT_PREFLIGHT_SMOKE_REPORT = Path(
    "docs_zh/mowa/mowa_e006_policy_rollout_preflight_smoke.json"
)
E006_POLICY_ROLLOUT_SMOKE_REPORT = Path("docs_zh/mowa/mowa_e006_policy_rollout_smoke.json")
CORE_SOT_DOCS = (
    "00_project_proposal.md",
    "01_technical_survey.md",
    "02_detailed_design.md",
)
DERIVED_DESIGN_DOCS = (
    "03_agent_implementation_plan.md",
    "04_task_breakdown.md",
    "05_experiment_registry.md",
    "06_data_gate_report.md",
    "07_implementation_log.md",
    "08_starvla_data_benchmark_support_matrix.md",
    "09_future_label_builder_design.md",
    "10_future_latent_cache_manifest_design.md",
)


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
    window_preflight = reports.get("production_window_preflight") or {}
    full_heads_smoke = reports.get("full_heads_train_smoke") or {}

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
        "full_heads_one_step_smoke_passed": (
            "train smoke passed" in str(full_heads_smoke.get("go_no_go", ""))
            and float(full_heads_smoke.get("loss_after", 1.0))
            <= float(full_heads_smoke.get("loss_before", 0.0))
        ),
        "production_window_5hz_preflight_passed": (
            window_preflight.get("split_status") == "smoke_passed"
            and window_preflight.get("worker_status") == "smoke_passed"
            and window_preflight.get("future_action_leakage_status") == "smoke_passed"
            and window_preflight.get("failed_sample_count") == 0
        ),
        "full_heads_interface_created": (root / FULL_HEADS_INTERFACE_CONFIG).is_file(),
        "e001_launch_draft_created": (root / E001_LAUNCH_DRAFT_CONFIG).is_file(),
        "e001_launch_draft_state_recorded": _launch_draft_state_recorded(
            root / E001_LAUNCH_DRAFT_CONFIG
        ),
        "runtime_policy_draft_created": (root / E001_RUNTIME_POLICY_DRAFT_CONFIG).is_file(),
        "runtime_policy_state_recorded": _runtime_policy_state_recorded(
            root / E001_RUNTIME_POLICY_DRAFT_CONFIG
        ),
        "action_bridge_interface_created": (root / MOWA_ACTION_BRIDGE_INTERFACE_CONFIG).is_file(),
        "training_command_draft_created": (root / E001_TRAINING_COMMAND_DRAFT_CONFIG).is_file(),
        "training_command_dry_run_only": _text_contains(
            root / E001_TRAINING_COMMAND_DRAFT_CONFIG,
            "dry_run_only: true",
        ),
        "training_command_candidate_created": (root / E001_TRAINING_COMMAND_CANDIDATE_CONFIG).is_file(),
        "launch_candidate_created": (root / E001_LAUNCH_CANDIDATE_CONFIG).is_file(),
        "training_command_candidate_state_consistent": _command_candidate_state_consistent(
            root / E001_TRAINING_COMMAND_CANDIDATE_CONFIG
        ),
        "launch_candidate_state_consistent": _launch_candidate_state_consistent(
            root / E001_LAUNCH_CANDIDATE_CONFIG,
            root / E001_RUNTIME_POLICY_DRAFT_CONFIG,
        ),
        "launch_candidate_smoke_passed": _checks_report_passed(
            root / E001_LAUNCH_CANDIDATE_SMOKE_REPORT
        ),
        "starflow_ft0_baseline_candidate_created": (
            root / E001_STARFLOW_FT0_BASELINE_CANDIDATE_CONFIG
        ).is_file(),
        "starflow_ft0_comparison_smoke_passed": _starflow_ft0_comparison_smoke_passed(
            root / E001_STARFLOW_FT0_COMPARISON_SMOKE_REPORT
        ),
        "training_smoke_config_created": (root / E001_TRAINING_SMOKE_CONFIG).is_file(),
        "training_config_smoke_passed": _training_config_smoke_passed(
            root / E001_TRAINING_CONFIG_SMOKE_REPORT
        ),
        "train_starvla_full_path_dry_run_config_created": (
            root / E001_TRAIN_STARVLA_DRY_RUN_CONFIG
        ).is_file(),
        "train_starvla_full_path_dry_run_smoke_passed": (
            _train_starvla_full_path_dry_run_smoke_passed(
                root / E001_TRAIN_STARVLA_DRY_RUN_SMOKE_REPORT
            )
        ),
        "starflow_ft0_full_path_dry_run_config_created": (
            root / E001_STARFLOW_FT0_DRY_RUN_CONFIG
        ).is_file(),
        "starflow_ft0_full_path_dry_run_smoke_passed": (
            _train_starvla_full_path_dry_run_smoke_passed(
                root / E001_STARFLOW_FT0_DRY_RUN_SMOKE_REPORT
            )
        ),
        "starflow_ft0_training_throughput_config_created": (
            root / E001_STARFLOW_FT0_TRAINING_THROUGHPUT_CONFIG
        ).is_file(),
        "starflow_ft0_training_throughput_smoke_passed": _checks_report_passed(
            root / E001_STARFLOW_FT0_TRAINING_THROUGHPUT_SMOKE_REPORT
        ),
        "a100_throughput_smoke_plan_created": (
            root / E001_A100_THROUGHPUT_SMOKE_PLAN_CONFIG
        ).is_file(),
        "a100_throughput_smoke_executed": _a100_throughput_smoke_passed(
            root / E001_A100_THROUGHPUT_SMOKE_REPORT
        ),
        "full_vla_runtime_sweep_bs4_passed": _checks_report_passed(
            root / E001_FULL_VLA_RUNTIME_SWEEP_REPORT
        ),
        "e006_coupling_intervention_smoke_passed": _checks_report_passed(
            root / E006_COUPLING_INTERVENTION_SMOKE_REPORT
        ),
        "e006_eval_load_smoke_config_created": (root / E006_EVAL_LOAD_SMOKE_CONFIG).is_file(),
        "e006_eval_load_smoke_passed": _checks_report_passed(
            root / E006_EVAL_LOAD_SMOKE_REPORT
        ),
        "e006_policy_rollout_preflight_smoke_passed": _checks_report_passed(
            root / E006_POLICY_ROLLOUT_PREFLIGHT_SMOKE_REPORT
        ),
        "e006_policy_rollout_outcome_recorded": _e006_rollout_outcome_recorded(
            root / E006_POLICY_ROLLOUT_SMOKE_REPORT
        ),
    }

    sot_status = _sot_docs_status(root)

    unresolved_items = []
    if missing_reports:
        unresolved_items.append(f"missing_reports={','.join(missing_reports)}")
    if sot_status["status"] != "available":
        unresolved_items.append(
            "core SOT docs missing="
            + ",".join(sot_status["missing_core_docs"])
        )
    unresolved_items.extend(
        [
            "E-001 launch draft records bounded approved state only, not full-scale launch approval",
            "runtime policy draft and launch candidate must stay synchronized",
            "batch size 4, expected VRAM and runtime are bounded-smoke observed only, not long-training confirmed",
            "E-001 executable command candidate exists; keep command/runtime/launch state synchronized",
            "E-006 rollout outcome is recorded in the rollout smoke report; action-gain evidence remains pending",
        ]
    )

    class_mapping_confirmed = str(full_heads_smoke.get("class_mapping_status")) != "Data Gate"
    ready_for_launch = (
        all(checks.values())
        and not missing_reports
        and sot_status["status"] == "available"
        and class_mapping_confirmed
    )
    launch_scope_status = _launch_scope_status(root / E001_LAUNCH_DRAFT_CONFIG)
    return {
        "stage": "full_heads",
        "experiment_id": "E-001",
        "experiment_name": "FullHeads vs VLA baseline",
        "training_started": False,
        "reports": {name: str(path) for name, path in REQUIRED_REPORTS.items()},
        "checks": checks,
        "sot": sot_status,
        "observed": {
            "task_count": recipe.get("task_count"),
            "available_task_count": recipe.get("available_task_count"),
            "train_episode_count": preflight.get("train_episode_count"),
            "val_episode_count": preflight.get("val_episode_count"),
            "split_overlap_count": preflight.get("split_overlap_count"),
            "distributed_overlap_count": preflight.get("distributed_overlap_count"),
            "worker_sample_count": preflight.get("worker_sample_count"),
            "failed_sample_count": preflight.get("failed_sample_count"),
            "production_wam_hz": 5,
            "raw_action_hz": 20,
            "wam_stride": 4,
            "history_steps": (window_preflight.get("window_config") or {}).get("history_steps"),
            "future_steps": (window_preflight.get("window_config") or {}).get("future_steps"),
            "action_chunk_steps": (window_preflight.get("window_config") or {}).get("action_chunk_steps"),
            "production_window_preflight_report": str(REQUIRED_REPORTS["production_window_preflight"]),
            "full_heads_smoke_sample_count": full_heads_smoke.get("sample_count"),
            "loss_before": full_heads_smoke.get("loss_before"),
            "loss_after": full_heads_smoke.get("loss_after"),
            "class_mapping_status": full_heads_smoke.get("class_mapping_status"),
            "full_heads_interface_config": str(FULL_HEADS_INTERFACE_CONFIG),
            "e001_launch_draft_config": str(E001_LAUNCH_DRAFT_CONFIG),
            "e001_runtime_policy_draft_config": str(E001_RUNTIME_POLICY_DRAFT_CONFIG),
            "mowa_action_bridge_interface_config": str(MOWA_ACTION_BRIDGE_INTERFACE_CONFIG),
            "e001_training_command_draft_config": str(E001_TRAINING_COMMAND_DRAFT_CONFIG),
            "e001_training_command_candidate_config": str(E001_TRAINING_COMMAND_CANDIDATE_CONFIG),
            "e001_launch_candidate_config": str(E001_LAUNCH_CANDIDATE_CONFIG),
            "e001_launch_candidate_smoke_report": str(E001_LAUNCH_CANDIDATE_SMOKE_REPORT),
            "e001_starflow_ft0_baseline_candidate_config": str(
                E001_STARFLOW_FT0_BASELINE_CANDIDATE_CONFIG
            ),
            "e001_starflow_ft0_comparison_smoke_report": str(
                E001_STARFLOW_FT0_COMPARISON_SMOKE_REPORT
            ),
            "e001_training_smoke_config": str(E001_TRAINING_SMOKE_CONFIG),
            "e001_training_config_smoke_report": str(E001_TRAINING_CONFIG_SMOKE_REPORT),
            "e001_train_starvla_full_path_dry_run_config": str(E001_TRAIN_STARVLA_DRY_RUN_CONFIG),
            "e001_train_starvla_full_path_dry_run_smoke_report": str(
                E001_TRAIN_STARVLA_DRY_RUN_SMOKE_REPORT
            ),
            "e001_starflow_ft0_full_path_dry_run_config": str(E001_STARFLOW_FT0_DRY_RUN_CONFIG),
            "e001_starflow_ft0_full_path_dry_run_smoke_report": str(
                E001_STARFLOW_FT0_DRY_RUN_SMOKE_REPORT
            ),
            "e001_starflow_ft0_training_throughput_config": str(
                E001_STARFLOW_FT0_TRAINING_THROUGHPUT_CONFIG
            ),
            "e001_starflow_ft0_training_throughput_smoke_report": str(
                E001_STARFLOW_FT0_TRAINING_THROUGHPUT_SMOKE_REPORT
            ),
            "e001_a100_throughput_smoke_plan_config": str(E001_A100_THROUGHPUT_SMOKE_PLAN_CONFIG),
            "e001_a100_throughput_smoke_report": str(E001_A100_THROUGHPUT_SMOKE_REPORT),
            "e006_coupling_intervention_smoke_report": str(E006_COUPLING_INTERVENTION_SMOKE_REPORT),
            "e006_eval_load_smoke_config": str(E006_EVAL_LOAD_SMOKE_CONFIG),
            "e006_eval_load_smoke_report": str(E006_EVAL_LOAD_SMOKE_REPORT),
            "e006_policy_rollout_preflight_smoke_report": str(
                E006_POLICY_ROLLOUT_PREFLIGHT_SMOKE_REPORT
            ),
            "e006_policy_rollout_smoke_report": str(E006_POLICY_ROLLOUT_SMOKE_REPORT),
            "e006_policy_rollout_blocker": (
                (_read_json(root / E006_POLICY_ROLLOUT_SMOKE_REPORT) or {}).get("rollout_blocker")
            ),
            "a100_throughput_stable_candidate": (
                (_read_json(root / E001_A100_THROUGHPUT_SMOKE_REPORT) or {}).get(
                    "stable_candidate"
                )
            ),
            "launch_scope_status": launch_scope_status,
        },
        "unresolved_items": unresolved_items,
        "go_no_go": (
            "TBD: E-001 prerequisites mostly passed; launch draft exists but runtime/batch/save-resume remain Data Gate"
            if ready_for_launch
            else "No-Go: E-001 prerequisites incomplete"
        ),
        "notes": [
            "This report does not start training and does not consume E-001.",
            "Existing QwenOFT RoboCasa365 scripts are baseline/walk-through scripts, not MoWA E-001 launch config.",
            "E-001 launch requires explicit user confirmation before modifying training entry points "
            "or checkpoint logic.",
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


def _sot_docs_status(root: Path) -> dict[str, Any]:
    docs_root = root / "docs_zh" / "mowa"
    available_core = tuple(name for name in CORE_SOT_DOCS if (docs_root / name).is_file())
    missing_core = tuple(name for name in CORE_SOT_DOCS if name not in available_core)
    available_derived = tuple(
        name for name in DERIVED_DESIGN_DOCS if (docs_root / name).is_file()
    )
    missing_derived = tuple(
        name for name in DERIVED_DESIGN_DOCS if name not in available_derived
    )
    return {
        "required_core_docs": CORE_SOT_DOCS,
        "available_core_docs": available_core,
        "missing_core_docs": missing_core,
        "derived_design_docs": DERIVED_DESIGN_DOCS,
        "available_derived_design_docs": available_derived,
        "missing_derived_design_docs": missing_derived,
        "status": "available" if not missing_core else "missing_core_sot",
        "note": (
            "Core SOT docs are available; treat them as the source of truth for governance "
            "and readiness."
            if not missing_core
            else "Do not recreate or locally rewrite missing core SOT docs; keep launch gated "
            "and record the gap in implementation logs."
        ),
    }


def _launch_scope_status(path: Path) -> str:
    payload = _read_yaml(path)
    launch = (payload or {}).get("launch") or {}
    if (
        launch.get("launch_ready") is True
        and launch.get("training_started") is True
        and launch.get("training_completed_1000_steps") is True
    ):
        return "bounded_approved_state_only"
    if launch.get("launch_ready") is True:
        return "launch_ready_state_recorded"
    return "launch_not_ready_or_missing"


def _a100_throughput_smoke_passed(path: Path) -> bool:
    payload = _read_json(path)
    if payload is None:
        return False
    return (
        payload.get("benchmark") == "a100_throughput_smoke"
        and payload.get("training_started") is False
        and payload.get("checkpoint_saved") is False
        and payload.get("stable_candidate") is not None
        and "A100" in str((payload.get("gpu") or {}).get("name", ""))
    )


def _runtime_policy_state_recorded(path: Path) -> bool:
    payload = _read_yaml(path)
    status = (payload or {}).get("status") or {}
    return (
        "policy_confirmed" in status
        and "launch_ready" in status
        and "resource_policy_confirmed" in status
    )


def _launch_draft_state_recorded(path: Path) -> bool:
    payload = _read_yaml(path)
    launch = (payload or {}).get("launch") or {}
    return (
        "launch_ready" in launch
        and "training_started" in launch
        and "requires_human_confirmation" in launch
        and "human_confirmed" in launch
    )


def _command_candidate_state_consistent(path: Path) -> bool:
    payload = _read_yaml(path)
    guard = (payload or {}).get("launch_guard") or {}
    return (
        "launch_ready" in guard
        and guard.get("requires_human_confirmation") is True
        and "policy_confirmed" in guard
    )


def _launch_candidate_state_consistent(candidate_path: Path, runtime_policy_path: Path) -> bool:
    candidate = _read_yaml(candidate_path)
    runtime_policy = _read_yaml(runtime_policy_path)
    candidate_guard = (candidate or {}).get("launch_guard") or {}
    runtime_status = (runtime_policy or {}).get("status") or {}
    if "launch_ready" not in candidate_guard or "policy_confirmed" not in candidate_guard:
        return False
    if candidate_guard.get("launch_ready") is True:
        if candidate_guard.get("policy_confirmed") is not True:
            return False
        if candidate_guard.get("requires_human_confirmation") is True:
            if candidate_guard.get("human_confirmed") is not True:
                return False
    else:
        if candidate_guard.get("policy_confirmed") is not False:
            return False
    return (
        runtime_status.get("launch_ready") == candidate_guard.get("launch_ready")
        and runtime_status.get("policy_confirmed") == candidate_guard.get("policy_confirmed")
    )


def _training_config_smoke_passed(path: Path) -> bool:
    payload = _read_json(path)
    if payload is None:
        return False
    return (
        payload.get("training_started") is False
        and payload.get("launch_ready") is False
        and all((payload.get("checks") or {}).values())
    )


def _train_starvla_full_path_dry_run_smoke_passed(path: Path) -> bool:
    payload = _read_json(path)
    if payload is None:
        return False
    checks = payload.get("checks") or {}
    return payload.get("training_started") is False and all(checks.values())


def _checks_report_passed(path: Path) -> bool:
    payload = _read_json(path)
    if payload is None:
        return False
    checks = payload.get("checks") or {}
    return bool(checks) and all(checks.values())


def _starflow_ft0_comparison_smoke_passed(path: Path) -> bool:
    payload = _read_json(path)
    if payload is None:
        return False
    checks = payload.get("checks") or {}
    execution = payload.get("launch_guard_execution") or {}
    return (
        bool(checks)
        and all(checks.values())
        and checks.get("baseline_launch_gated") is True
        and checks.get("mowa_launch_state_consistent") is True
        and (
            execution.get("checked") is False
            or bool((execution.get("baseline") or {}).get("blocked_by_launch_guard"))
        )
    )


def _e006_rollout_outcome_recorded(path: Path) -> bool:
    payload = _read_json(path)
    if payload is None:
        return False
    blocker = payload.get("rollout_blocker") or {}
    if (
        blocker.get("status") in {
            "missing_robocasa_assets",
            "checkpoint_model_incompatible",
            "robocasa_render_backend_unavailable",
        }
        and blocker.get("scope") in {"environment", "checkpoint"}
        and bool(blocker.get("blocked_interventions"))
        and bool(
            blocker.get("missing_asset_paths")
            or blocker.get("missing_state_keys")
            or blocker.get("backend_signatures")
        )
    ):
        return True
    checks = payload.get("checks") or {}
    return (
        bool(checks)
        and checks.get("server_started_when_executed") is True
        and checks.get("client_succeeded_when_executed") is True
        and checks.get("result_json_collected_when_executed") is True
        and checks.get("success_rate_recorded_when_executed") is True
    )


def _read_yaml(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        from omegaconf import OmegaConf
    except ModuleNotFoundError:
        return None
    payload = OmegaConf.load(path)
    return OmegaConf.to_container(payload, resolve=True)


if __name__ == "__main__":
    main()
