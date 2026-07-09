"""Build a launch-readiness matrix for all MoWA experiments from existing reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


OUTPUT = Path("docs_zh/mowa/mowa_experiment_launch_readiness_matrix.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MoWA experiment launch readiness matrix.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_experiment_launch_readiness_matrix(args.repo_root)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def build_experiment_launch_readiness_matrix(repo_root: Path | str) -> dict[str, Any]:
    root = Path(repo_root)
    entries = [
        _build_g0_entry(root),
        _build_e001_entry(root),
        _build_e002_entry(root),
        _build_e003_entry(root),
        _build_e004_entry(root),
        _build_e005_entry(root),
        _build_e006_entry(root),
        _build_e007_entry(root),
        _build_e008_entry(root),
        _build_e009_entry(root),
        _build_e010_entry(root),
    ]
    training_entries = [entry for entry in entries if entry["counts_as_training_experiment"]]
    launchable_now = [
        entry["experiment_id"]
        for entry in training_entries
        if entry["can_start_now"] is True
    ]
    return {
        "project": "MoWA",
        "generated_from": "existing smoke/readiness reports only",
        "training_started": False,
        "all_training_experiments_ready": len(launchable_now) == len(training_entries),
        "ready_training_experiments": launchable_now,
        "not_ready_training_experiments": [
            entry["experiment_id"]
            for entry in training_entries
            if entry["can_start_now"] is not True
        ],
        "entries": entries,
        "global_blockers": _global_blockers(entries),
        "go_no_go": (
            "TBD: launch readiness matrix generated; all active training experiments are launch-ready"
            if len(launchable_now) == len(training_entries)
            else "No-Go: full experiment suite is not launch-ready"
        ),
    }


def _build_g0_entry(root: Path) -> dict[str, Any]:
    recipe = _load_json(root, "docs_zh/mowa/mowa_g0_robocasa365_atomic_core_recipe_smoke.json")
    preflight = _load_json(
        root,
        "docs_zh/mowa/g0_atomic_core_smoke/mowa_g0_atomic_core_production_preflight_smoke.json",
    )
    window = _load_json(
        root,
        "docs_zh/mowa/g0_atomic_core_smoke/mowa_g0_atomic_core_production_window_5hz_preflight_smoke.json",
    )
    leakage = _load_json(
        root,
        "docs_zh/mowa/g0_atomic_core_smoke/mowa_g0_atomic_core_leakage_gate_smoke.json",
    )
    label_builder = _load_json(
        root,
        "docs_zh/mowa/g0_atomic_core_smoke/mowa_g0_atomic_core_future_label_builder_summary.json",
    )
    checks = {
        "recipe_ready": _report_not_nogo(recipe),
        "production_preflight_ready": _report_not_nogo(preflight),
        "production_window_ready": _report_not_nogo(window),
        "leakage_gate_ready": _report_not_nogo(leakage),
        "constructible_label_smoke_ready": _report_not_nogo(label_builder),
    }
    status = (
        "bounded_data_domain_ready_for_following_experiments"
        if all(checks.values())
        else "data_gate_incomplete"
    )
    return {
        "experiment_id": "G0",
        "stage": "Gate",
        "counts_as_training_experiment": False,
        "can_start_now": False,
        "status": status,
        "evidence_reports": _present_reports(
            {
                "recipe": recipe,
                "production_preflight": preflight,
                "production_window_preflight": window,
                "leakage_gate": leakage,
                "constructible_label_builder": label_builder,
            }
        ),
        "blocking_items": [
            "G0 is a gate, not a launchable experiment.",
            "Current data-domain evidence is bounded to RoboCasa365 target/human atomic core.",
        ],
        "checks": checks,
    }


def _build_e001_entry(root: Path) -> dict[str, Any]:
    readiness = _load_json(root, "docs_zh/mowa/mowa_e001_readiness_smoke.json")
    launch_candidate = _load_json(root, "docs_zh/mowa/mowa_e001_launch_candidate_smoke.json")
    comparison = _load_json(root, "docs_zh/mowa/mowa_e001_starflow_ft0_comparison_smoke.json")
    long_training = _load_yaml(
        root,
        "configs/mowa/mowa_e001_starflow_ft0_long_training_candidate.yaml",
    )
    baseline_long_training = _load_yaml(
        root,
        "configs/mowa/mowa_e001_starflow_ft0_baseline_long_training_candidate.yaml",
    )
    launch_ready = bool((launch_candidate.get("launch_guard") or {}).get("launch_ready"))
    long_training_ready = _launch_guard_ready(long_training) and _launch_guard_ready(
        baseline_long_training
    )
    readiness_nogo = str(readiness.get("go_no_go", "")).startswith("No-Go")
    status = (
        "launchable_now"
        if launch_ready and long_training_ready
        else "bounded_executable_but_full_launch_blocked"
        if launch_ready and readiness_nogo
        else "training_path_incomplete"
    )
    blockers = list(readiness.get("unresolved_items") or [])
    return {
        "experiment_id": "E-001",
        "stage": "full_heads",
        "counts_as_training_experiment": True,
        "can_start_now": bool(launch_ready and long_training_ready),
        "status": status,
        "evidence_reports": _present_reports(
            {
                "readiness": readiness,
                "launch_candidate": launch_candidate,
                "paired_comparison": comparison,
                "long_training_candidate": long_training,
                "baseline_long_training_candidate": baseline_long_training,
            }
        ),
        "blocking_items": blockers,
        "checks": {
            "launch_candidate_launch_ready": launch_ready,
            "long_training_candidate_launch_ready": _launch_guard_ready(long_training),
            "baseline_long_training_candidate_launch_ready": _launch_guard_ready(
                baseline_long_training
            ),
            "readiness_is_not_nogo": not readiness_nogo,
            "paired_comparison_present": _report_exists(comparison),
        },
    }


def _build_e002_entry(root: Path) -> dict[str, Any]:
    interface = _load_json(root, "docs_zh/mowa/mowa_future_gated_heads_interface_smoke.json")
    comparison = _load_json(root, "docs_zh/mowa/mowa_e002_future_gated_heads_comparison_smoke.json")
    comparison_ready = _report_not_nogo(comparison) and bool(
        (comparison.get("checks") or {}).get("candidate_launch_guard_open")
    )
    return {
        "experiment_id": "E-002",
        "stage": "full_heads",
        "counts_as_training_experiment": True,
        "can_start_now": bool(_report_not_nogo(interface) and comparison_ready),
        "status": (
            "launchable_now"
            if _report_not_nogo(interface) and comparison_ready
            else "comparison_entry_incomplete"
        ),
        "evidence_reports": _present_reports(
            {
                "future_gated_heads_interface": interface,
                "single_comparison_entry": comparison,
            }
        ),
        "blocking_items": list(comparison.get("unresolved_items") or []),
        "checks": {
            "interface_smoke_passed": _report_not_nogo(interface),
            "single_comparison_entry_present": _report_not_nogo(comparison),
            "candidate_launch_guard_open": bool(
                (comparison.get("checks") or {}).get("candidate_launch_guard_open")
            ),
        },
    }


def _build_e003_entry(root: Path) -> dict[str, Any]:
    prior = _load_json(root, "docs_zh/mowa/mowa_future_latent_prior_interface_smoke.json")
    builder = _load_json(root, "docs_zh/mowa/mowa_latent_cache_builder_design_smoke.json")
    consistency = _load_json(root, "docs_zh/mowa/mowa_e003_history_sampling_consistency_smoke.json")
    config_preview = _load_json(root, "docs_zh/mowa/mowa_e003_future_latent_prior_config_preview.json")
    cache_dry_run = _load_json(root, "docs_zh/mowa/mowa_e003_future_latent_prior_train_dry_run.json")
    launch_smoke = _load_json(root, "docs_zh/mowa/mowa_e003_future_latent_prior_launch_smoke.json")
    wanpi_dry_run = _load_json(root, "docs_zh/mowa/mowa_e003_wanpi_future_latent_prior_dry_run.json")
    blockers = _merged_unresolved(prior, builder, consistency, config_preview, cache_dry_run)
    blockers.extend(
        [
            "E-003 has been migrated from the StarFlowVLA/QwenPI_v3 path to the WanPI path; "
            "a WanPI-MoWA full-path dry-run on the production episode latent cache is required before launch.",
        ]
    )
    if not _report_not_nogo(launch_smoke):
        blockers.append("E-003 launch smoke has not yet passed on the real Wan2.2 cache path.")
    if not _report_not_nogo(wanpi_dry_run):
        blockers.append("E-003 WanPI dry-run evidence is not yet available.")
    preconditions_met = (
        _report_not_nogo(prior)
        and _report_not_nogo(builder)
        and _report_not_nogo(consistency)
        and _report_not_nogo(config_preview)
        and _report_not_nogo(cache_dry_run)
        and _report_not_nogo(launch_smoke)
    )
    return {
        "experiment_id": "E-003",
        "stage": "future_latent_prior",
        "counts_as_training_experiment": True,
        "can_start_now": bool(preconditions_met),
        "status": (
            "launchable_now"
            if preconditions_met
            else "future_latent_prior_preconditions_incomplete"
        ),
        "evidence_reports": _present_reports(
            {
                "future_latent_prior_interface": prior,
                "latent_cache_builder_design": builder,
                "history_sampling_consistency": consistency,
                "future_latent_prior_config_preview": config_preview,
                "future_latent_prior_train_dry_run": cache_dry_run,
                "future_latent_prior_launch_smoke": launch_smoke,
                "wanpi_future_latent_prior_dry_run": wanpi_dry_run,
            }
        ),
        "blocking_items": blockers,
        "checks": {
            "future_latent_prior_interface_passed": _report_not_nogo(prior),
            "latent_cache_builder_design_passed": _report_not_nogo(builder),
            "history_sampling_consistency_passed": _report_not_nogo(consistency),
            "future_latent_prior_config_preview_passed": _report_not_nogo(config_preview),
            "future_latent_prior_train_dry_run_passed": _report_not_nogo(cache_dry_run),
            "future_latent_prior_launch_smoke_passed": _report_not_nogo(launch_smoke),
            "wanpi_future_latent_prior_dry_run_passed": _report_not_nogo(wanpi_dry_run),
        },
    }


def _build_e004_entry(root: Path) -> dict[str, Any]:
    config_preview = _load_json(root, "docs_zh/mowa/mowa_e004_hlc_gci_config_preview.json")
    hlcgci = _load_json(root, "docs_zh/mowa/mowa_hlc_gci_interface_smoke.json")
    consistency = _load_json(root, "docs_zh/mowa/mowa_e003_history_sampling_consistency_smoke.json")
    launch_smoke = _load_json(root, "docs_zh/mowa/mowa_e004_hlc_gci_launch_smoke.json")
    checkpoint_preflight = _load_json(
        root,
        "docs_zh/mowa/mowa_e004_hlc_gci_checkpoint_preflight_smoke.json",
    )
    policy_rollout = _load_json(root, "docs_zh/mowa/mowa_e004_hlc_gci_policy_rollout_smoke.json")
    wanpi_dry_run = _load_json(root, "docs_zh/mowa/mowa_e004_wanpi_hlc_gci_dry_run.json")
    blockers = _merged_unresolved(
        config_preview,
        hlcgci,
        consistency,
        launch_smoke,
        checkpoint_preflight,
        policy_rollout,
    )
    blockers.extend(
        [
            "E-004 has been migrated from the StarFlowVLA/QwenPI_v3 path to the WanPI path; "
            "a WanPI-MoWA full-path dry-run on the production episode latent cache is required before launch.",
            "E-004 rollout remains gated on a trained E-003/E-004 checkpoint and success metrics.",
        ]
    )
    if not _report_not_nogo(launch_smoke):
        blockers.append("E-004 launch smoke has not yet passed on the synthetic HLC-GCI path.")
    if not _report_not_nogo(wanpi_dry_run):
        blockers.append("E-004 WanPI dry-run evidence is not yet available.")
    if not _report_not_nogo(checkpoint_preflight):
        blockers.append("E-004 checkpoint-backed preflight has not yet passed on a complete checkpoint reference.")
    if not _report_not_nogo(policy_rollout):
        blockers.append("E-004 checkpoint-backed policy rollout smoke has not yet passed.")
    preconditions_met = (
        _report_not_nogo(config_preview)
        and _report_not_nogo(hlcgci)
        and _report_not_nogo(consistency)
        and _report_not_nogo(launch_smoke)
        and _report_not_nogo(checkpoint_preflight)
        and _report_not_nogo(policy_rollout)
    )
    return {
        "experiment_id": "E-004",
        "stage": "hlc_gci",
        "counts_as_training_experiment": True,
        "can_start_now": bool(preconditions_met),
        "status": (
            "launchable_now"
            if preconditions_met
            else "hlc_gci_preconditions_incomplete"
        ),
        "evidence_reports": _present_reports(
            {
                "hlcgci_config_preview": config_preview,
                "hlcgci_interface": hlcgci,
                "history_sampling_consistency": consistency,
                "hlcgci_launch_smoke": launch_smoke,
                "hlcgci_checkpoint_preflight": checkpoint_preflight,
                "hlcgci_policy_rollout": policy_rollout,
                "wanpi_hlc_gci_dry_run": wanpi_dry_run,
            }
        ),
        "blocking_items": blockers,
        "checks": {
            "hlcgci_config_preview_passed": _report_not_nogo(config_preview),
            "hlcgci_interface_passed": _report_not_nogo(hlcgci),
            "history_sampling_consistency_passed": _report_not_nogo(consistency),
            "hlcgci_launch_smoke_passed": _report_not_nogo(launch_smoke),
            "hlcgci_checkpoint_preflight_passed": _report_not_nogo(checkpoint_preflight),
            "hlcgci_policy_rollout_passed": _report_not_nogo(policy_rollout),
            "wanpi_hlc_gci_dry_run_passed": _report_not_nogo(wanpi_dry_run),
            "hlcgci_policy_rollout_zero_success": (
                bool(policy_rollout)
                and all(
                    (run.get("rollout_result") or {}).get("success_rate") == 0.0
                    for run in (policy_rollout.get("runs") or [])
                )
            ),
        },
    }


def _build_e005_entry(root: Path) -> dict[str, Any]:
    shuffled = _load_json(root, "docs_zh/mowa/mowa_shuffled_robot_sanity_plan_smoke.json")
    checkpoint_preflight = _load_json(
        root,
        "docs_zh/mowa/mowa_e005_shuffled_robot_checkpoint_preflight_smoke.json",
    )
    blockers = list(shuffled.get("unresolved_items") or [])
    blockers.extend(list(checkpoint_preflight.get("unresolved_items") or []))
    blockers.extend(
        [
            "E-005 requires a meaningful E-004 checkpoint; shuffled-robot plan/smoke alone is not execution evidence.",
            "Metric-drop expectation must be validated after P1-b1 training and cannot be claimed from pair construction.",
        ]
    )
    preflight_ready = _report_not_nogo(checkpoint_preflight)
    return {
        "experiment_id": "E-005",
        "stage": "hlc_gci",
        "counts_as_training_experiment": True,
        "can_start_now": preflight_ready,
        "status": (
            "launchable_now"
            if preflight_ready
            else "plan_ready_only"
            if _report_not_nogo(shuffled)
            else "sanity_plan_missing"
        ),
        "evidence_reports": _present_reports(
            {
                "shuffled_robot_sanity_plan": shuffled,
                "shuffled_robot_checkpoint_preflight": checkpoint_preflight,
            }
        ),
        "blocking_items": blockers,
        "checks": {
            "sanity_plan_present": _report_not_nogo(shuffled),
            "checkpoint_preflight_passed": _report_not_nogo(checkpoint_preflight),
        },
    }


def _build_e006_entry(root: Path) -> dict[str, Any]:
    eval_load = _load_json(root, "docs_zh/mowa/mowa_e006_eval_load_smoke.json")
    synthetic = _load_json(root, "docs_zh/mowa/mowa_e006_coupling_intervention_smoke.json")
    checkpoint_forward = _load_json(
        root,
        "docs_zh/mowa/mowa_e006_checkpoint_intervention_forward_smoke.json",
    )
    rollout_preflight = _load_json(root, "docs_zh/mowa/mowa_e006_policy_rollout_preflight_smoke.json")
    rollout = _load_json(root, "docs_zh/mowa/mowa_e006_policy_rollout_smoke.json")
    checks = {
        "eval_load_passed": _report_not_nogo(eval_load),
        "synthetic_intervention_passed": _report_not_nogo(synthetic),
        "checkpoint_forward_passed": _report_not_nogo(checkpoint_forward),
        "rollout_preflight_passed": _report_not_nogo(rollout_preflight),
        "rollout_executed": bool(rollout) and not str(rollout.get("go_no_go", "")).startswith("No-Go"),
    }
    launchable = (
        checks["eval_load_passed"]
        and checks["synthetic_intervention_passed"]
        and checks["rollout_preflight_passed"]
    )
    return {
        "experiment_id": "E-006",
        "stage": "full_heads/future_latent_prior",
        "counts_as_training_experiment": True,
        "can_start_now": launchable,
        "status": "launchable_now" if launchable else "coupling_evidence_incomplete",
        "evidence_reports": _present_reports(
            {
                "eval_load": eval_load,
                "synthetic_intervention": synthetic,
                "checkpoint_forward": checkpoint_forward,
                "rollout_preflight": rollout_preflight,
                "rollout": rollout,
            }
        ),
        "blocking_items": _merged_unresolved(
            eval_load,
            synthetic,
            checkpoint_forward,
            rollout_preflight,
            rollout,
        ),
        "checks": checks,
    }


def _build_e007_entry(root: Path) -> dict[str, Any]:
    readiness = _load_json(root, "docs_zh/mowa/mowa_e007_proxy_alpha_readiness_smoke.json")
    return {
        "experiment_id": "E-007",
        "stage": "Data mix",
        "counts_as_training_experiment": True,
        "can_start_now": _report_not_nogo(readiness) and bool(readiness.get("launch_ready")),
        "status": (
            "launchable_now"
            if _report_not_nogo(readiness) and bool(readiness.get("launch_ready"))
            else "proxy_alpha_candidate_incomplete"
        ),
        "evidence_reports": _present_reports({"proxy_alpha_readiness": readiness}),
        "blocking_items": list(readiness.get("unresolved_items") or []),
        "checks": {
            "readiness_report_present": _report_exists(readiness),
            "proxy_alpha_readiness_passed": _report_not_nogo(readiness),
        },
    }


def _build_e008_entry(root: Path) -> dict[str, Any]:
    diagnostic = _load_json(root, "docs_zh/mowa/mowa_frozen_decoder_diagnostic_smoke.json")
    return {
        "experiment_id": "E-008",
        "stage": "frozen_decoder_diagnostic",
        "counts_as_training_experiment": False,
        "can_start_now": False,
        "status": "diagnostic_plan_ready_only" if _report_not_nogo(diagnostic) else "diagnostic_plan_missing",
        "evidence_reports": _present_reports({"frozen_decoder_diagnostic": diagnostic}),
        "blocking_items": list(diagnostic.get("unresolved_items") or []),
        "checks": {
            "diagnostic_plan_present": _report_not_nogo(diagnostic),
        },
    }


def _build_e009_entry(root: Path) -> dict[str, Any]:
    return {
        "experiment_id": "E-009",
        "stage": "latent_cache",
        "counts_as_training_experiment": False,
        "can_start_now": False,
        "status": "conditional_not_triggered",
        "evidence_reports": [],
        "blocking_items": [
            "E-009 is conditional; it is not triggered while short-window HLC-GCI remains the active P1-b1 path."
        ],
        "checks": {
            "conditional_triggered": False,
        },
    }


def _build_e010_entry(root: Path) -> dict[str, Any]:
    tracking = _load_json(root, "docs_zh/mowa/mowa_e010_eval_tracking_smoke.json")
    return {
        "experiment_id": "E-010",
        "stage": "Eval-only",
        "counts_as_training_experiment": False,
        "can_start_now": _report_not_nogo(tracking) and bool(tracking.get("launch_ready")),
        "status": (
            "launchable_now"
            if _report_not_nogo(tracking) and bool(tracking.get("launch_ready"))
            else "eval_tracking_not_instantiated"
        ),
        "evidence_reports": _present_reports({"eval_tracking": tracking}),
        "blocking_items": list(tracking.get("unresolved_items") or []),
        "checks": {
            "tracking_report_present": _report_exists(tracking),
            "tracking_plan_passed": _report_not_nogo(tracking),
        },
    }


def _not_started_entry(root: Path, *, experiment_id: str, stage: str, message: str) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "stage": stage,
        "counts_as_training_experiment": True,
        "can_start_now": False,
        "status": "not_started",
        "evidence_reports": [],
        "blocking_items": [message],
        "checks": {
            "readiness_report_present": False,
        },
    }


def _global_blockers(entries: list[dict[str, Any]]) -> list[str]:
    blockers: list[str] = []
    for entry in entries:
        if entry["counts_as_training_experiment"] and not entry["can_start_now"]:
            first = (entry.get("blocking_items") or ["missing readiness evidence"])[0]
            blockers.append(f"{entry['experiment_id']}: {first}")
    return blockers


def _load_json(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    if not path.is_file():
        return {}
    try:
        import yaml
    except ModuleNotFoundError:
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _present_reports(reports: dict[str, dict[str, Any]]) -> list[str]:
    return [name for name, payload in reports.items() if _report_exists(payload)]


def _report_exists(payload: dict[str, Any]) -> bool:
    return bool(payload)


def _launch_guard_ready(payload: dict[str, Any]) -> bool:
    guard = payload.get("launch_guard") or {}
    return (
        guard.get("launch_ready") is True
        and guard.get("policy_confirmed") is True
        and (
            guard.get("requires_human_confirmation") is not True
            or guard.get("human_confirmed") is True
        )
    )


def _report_not_nogo(payload: dict[str, Any]) -> bool:
    if not payload:
        return False
    return not str(payload.get("go_no_go", "")).startswith("No-Go")


def _merged_unresolved(*payloads: dict[str, Any]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for payload in payloads:
        for item in payload.get("unresolved_items") or []:
            if item in seen:
                continue
            merged.append(item)
            seen.add(item)
    return merged


if __name__ == "__main__":
    main()
