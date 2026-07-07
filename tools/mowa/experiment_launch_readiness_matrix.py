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
            "TBD: launch readiness matrix generated; full experiment suite is not launch-ready"
            if launchable_now
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
    launch_ready = bool((launch_candidate.get("launch_guard") or {}).get("launch_ready"))
    readiness_nogo = str(readiness.get("go_no_go", "")).startswith("No-Go")
    status = "bounded_executable_but_full_launch_blocked" if launch_ready and readiness_nogo else (
        "launchable_now" if launch_ready and not readiness_nogo else "training_path_incomplete"
    )
    blockers = list(readiness.get("unresolved_items") or [])
    return {
        "experiment_id": "E-001",
        "stage": "full_heads",
        "counts_as_training_experiment": True,
        "can_start_now": bool(launch_ready and not readiness_nogo),
        "status": status,
        "evidence_reports": _present_reports(
            {
                "readiness": readiness,
                "launch_candidate": launch_candidate,
                "paired_comparison": comparison,
            }
        ),
        "blocking_items": blockers,
        "checks": {
            "launch_candidate_launch_ready": launch_ready,
            "readiness_is_not_nogo": not readiness_nogo,
            "paired_comparison_present": _report_exists(comparison),
        },
    }


def _build_e002_entry(root: Path) -> dict[str, Any]:
    interface = _load_json(root, "docs_zh/mowa/mowa_future_gated_heads_interface_smoke.json")
    comparison = _load_json(root, "docs_zh/mowa/mowa_e002_future_gated_heads_comparison_smoke.json")
    return {
        "experiment_id": "E-002",
        "stage": "full_heads",
        "counts_as_training_experiment": True,
        "can_start_now": False,
        "status": "runtime_integrated_but_training_gated"
        if _report_not_nogo(interface) and _report_not_nogo(comparison)
        else "comparison_entry_incomplete",
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
            "launch_guard_kept_closed": bool(
                (comparison.get("checks") or {}).get("candidate_is_launch_gated")
            ),
        },
    }


def _build_e003_entry(root: Path) -> dict[str, Any]:
    prior = _load_json(root, "docs_zh/mowa/mowa_future_latent_prior_interface_smoke.json")
    builder = _load_json(root, "docs_zh/mowa/mowa_latent_cache_builder_design_smoke.json")
    consistency = _load_json(root, "docs_zh/mowa/mowa_e003_history_sampling_consistency_smoke.json")
    blockers = _merged_unresolved(prior, builder, consistency)
    blockers.extend(
        [
            "Next required step is a fake-encoder latent cache writer/validator/loader loop; "
            "plan-only manifest and contract smoke are not sufficient for E-003 dry-run.",
            "E-003 must add config preview and train dry-run on real cache batches before any formal training launch.",
        ]
    )
    return {
        "experiment_id": "E-003",
        "stage": "future_latent_prior",
        "counts_as_training_experiment": True,
        "can_start_now": False,
        "status": "interface_ready_but_latent_cache_builder_missing"
        if _report_not_nogo(prior) and _report_not_nogo(builder) and _report_not_nogo(consistency)
        else "future_latent_prior_preconditions_incomplete",
        "evidence_reports": _present_reports(
            {
                "future_latent_prior_interface": prior,
                "latent_cache_builder_design": builder,
                "history_sampling_consistency": consistency,
            }
        ),
        "blocking_items": blockers,
        "checks": {
            "future_latent_prior_interface_passed": _report_not_nogo(prior),
            "latent_cache_builder_design_passed": _report_not_nogo(builder),
            "history_sampling_consistency_passed": _report_not_nogo(consistency),
        },
    }


def _build_e004_entry(root: Path) -> dict[str, Any]:
    hlcgci = _load_json(root, "docs_zh/mowa/mowa_hlc_gci_interface_smoke.json")
    consistency = _load_json(root, "docs_zh/mowa/mowa_e003_history_sampling_consistency_smoke.json")
    blockers = _merged_unresolved(hlcgci, consistency)
    blockers.extend(
        [
            "E-004 is blocked on E-003 fake-encoder cache loop and E-003 config preview/dry-run evidence.",
            "HLC-GCI interface alone is insufficient; history latent cache and framework integration remain missing.",
        ]
    )
    return {
        "experiment_id": "E-004",
        "stage": "hlc_gci",
        "counts_as_training_experiment": True,
        "can_start_now": False,
        "status": "module_interface_ready_but_framework_integration_missing"
        if _report_not_nogo(hlcgci) and _report_not_nogo(consistency)
        else "hlc_gci_preconditions_incomplete",
        "evidence_reports": _present_reports(
            {
                "hlcgci_interface": hlcgci,
                "history_sampling_consistency": consistency,
            }
        ),
        "blocking_items": blockers,
        "checks": {
            "hlcgci_interface_passed": _report_not_nogo(hlcgci),
            "history_sampling_consistency_passed": _report_not_nogo(consistency),
        },
    }


def _build_e005_entry(root: Path) -> dict[str, Any]:
    shuffled = _load_json(root, "docs_zh/mowa/mowa_shuffled_robot_sanity_plan_smoke.json")
    blockers = list(shuffled.get("unresolved_items") or [])
    blockers.extend(
        [
            "E-005 requires a meaningful E-004 checkpoint; shuffled-robot plan/smoke alone is not execution evidence.",
            "Metric-drop expectation must be validated after P1-b1 training and cannot be claimed from pair construction.",
        ]
    )
    return {
        "experiment_id": "E-005",
        "stage": "hlc_gci",
        "counts_as_training_experiment": True,
        "can_start_now": False,
        "status": "plan_ready_only" if _report_not_nogo(shuffled) else "sanity_plan_missing",
        "evidence_reports": _present_reports({"shuffled_robot_sanity_plan": shuffled}),
        "blocking_items": blockers,
        "checks": {
            "sanity_plan_present": _report_not_nogo(shuffled),
        },
    }


def _build_e006_entry(root: Path) -> dict[str, Any]:
    return {
        "experiment_id": "E-006",
        "stage": "full_heads/future_latent_prior",
        "counts_as_training_experiment": True,
        "can_start_now": False,
        "status": "coupling_evidence_incomplete",
        "evidence_reports": [],
        "blocking_items": [
            "E-006 coupling evidence reports reference deleted checkpoints and have been cleared. "
            "Regenerate eval-load, intervention, forward, preflight and rollout reports with a "
            "meaningful trained checkpoint before claiming coupling gain."
        ],
        "checks": {
            "eval_load_passed": False,
            "synthetic_intervention_passed": False,
            "checkpoint_forward_passed": False,
            "rollout_preflight_passed": False,
            "rollout_executed": False,
        },
    }


def _build_e007_entry(root: Path) -> dict[str, Any]:
    return _not_started_entry(
        root,
        experiment_id="E-007",
        stage="Data mix",
        message="No proxy-learned alpha readiness report exists yet.",
    )


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
    return _not_started_entry(
        root,
        experiment_id="E-009",
        stage="latent_cache",
        message="Conditional experiment has not been triggered and has no readiness artifacts.",
    )


def _build_e010_entry(root: Path) -> dict[str, Any]:
    return {
        "experiment_id": "E-010",
        "stage": "Eval-only",
        "counts_as_training_experiment": False,
        "can_start_now": False,
        "status": "eval_tracking_not_instantiated",
        "evidence_reports": [],
        "blocking_items": [
            "Eval-only tracking depends on having completed checkpoints from earlier experiments.",
            "No multi-benchmark tracking report exists yet.",
        ],
        "checks": {
            "tracking_report_present": False,
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


def _present_reports(reports: dict[str, dict[str, Any]]) -> list[str]:
    return [name for name, payload in reports.items() if _report_exists(payload)]


def _report_exists(payload: dict[str, Any]) -> bool:
    return bool(payload)


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
