"""Validate MoWA E-001 StarFlow full-path dry-run output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from starVLA.mowa_constants import (
    MOWA_FUTURE_FEATURE_SOURCE_ALIASES,
    MOWA_STARFLOW_CONDITION_PROBE_FEATURE_SOURCE,
)

DRY_RUN_CONFIG = Path("configs/mowa/mowa_e001_starflow_ft0_full_path_dry_run.yaml")
DRY_RUN_REPORT = Path("docs_zh/mowa/mowa_e001_starflow_ft0_full_path_dry_run.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoWA E-001 StarFlow dry-run report smoke.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_starflow_ft0_full_path_dry_run_smoke(args.repo_root)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def build_starflow_ft0_full_path_dry_run_smoke(repo_root: Path | str) -> dict[str, Any]:
    root = Path(repo_root)
    report = _read_json(root / DRY_RUN_REPORT) or {}
    framework = report.get("framework") or {}
    data = report.get("data") or {}
    batch_summary = data.get("batch_summary") or {}
    forward = report.get("forward") or {}
    forward_keys = forward.get("keys") or []
    first_item_keys = batch_summary.get("first_item_keys") or []
    checks = {
        "dry_run_config_created": (root / DRY_RUN_CONFIG).is_file(),
        "dry_run_report_created": (root / DRY_RUN_REPORT).is_file(),
        "entrypoint_is_train_starvla": report.get("entrypoint") == "starVLA/training/train_starvla.py",
        "full_path_dry_run_only": report.get("full_path_dry_run_only") is True,
        "training_not_started": report.get("training_started") is False,
        "checkpoint_not_saved": report.get("checkpoint_saved") is False,
        "wandb_not_started": report.get("wandb_started") is False,
        "framework_starflow": framework.get("name") == "StarFlowVLA",
        "action_head_layerwisefm": framework.get("action_model_type") == "LayerwiseFM",
        "starflow_ft_variant_configured": _starflow_ft_variant_matches_tokens(framework),
        "robocasa_data_mix": data.get("data_mix") == "robocasa365_open_drawer_target_human",
        "state_included": "state" in first_item_keys,
        "mowa_p0_labels_enabled": data.get("mowa_p0_labels_enabled") is True,
        "batch_has_mowa_p0_targets": "mowa_p0_targets" in first_item_keys,
        "batch_has_mowa_p0_masks": "mowa_p0_masks" in first_item_keys,
        "forward_evaluated": forward.get("evaluated") is True,
        "forward_has_action_loss": "action_loss" in forward_keys,
        "forward_metric_scope_one_batch": (
            forward.get("metric_scope") == "one_batch_no_backward_forward_dry_run"
        ),
        "forward_has_elapsed_sec": _is_positive_number(forward.get("elapsed_sec")),
        "forward_has_samples_per_sec": _is_positive_number(forward.get("samples_per_sec")),
        "forward_has_allocated_vram_field": "allocated_vram_gb" in forward,
        "forward_has_reserved_vram_field": "reserved_vram_gb" in forward,
        "forward_has_peak_vram_field": "peak_vram_gb" in forward,
        "forward_has_peak_reserved_vram_field": "peak_reserved_vram_gb" in forward,
        "forward_vram_metric_scope_recorded": (
            forward.get("vram_metric_scope") == "torch_cuda_allocator_in_full_path_dry_run"
        ),
        "mowa_layerwise_bridge_coupling_enabled": (
            framework.get("mowa_layerwise_bridge_coupling_enabled") is True
        ),
        "mowa_layerwise_bridge_forward_coupled": (
            framework.get("mowa_layerwise_bridge_coupling_status")
            == "forward_coupled_in_full_path_dry_run"
        ),
        "forward_has_mowa_layerwise_bridge_coupled": (
            "mowa_layerwise_bridge_coupled" in forward_keys
        ),
        "mowa_layerwise_bridge_uses_future_feature_heads": (
            forward.get("mowa_layerwise_bridge_feature_source") in MOWA_FUTURE_FEATURE_SOURCE_ALIASES
            and framework.get("mowa_layerwise_bridge_feature_source") in MOWA_FUTURE_FEATURE_SOURCE_ALIASES
        ),
        "mowa_layerwise_bridge_active_heads_are_p0": (
            forward.get("mowa_layerwise_bridge_active_heads")
            == ["task_progress", "action_outcome_class"]
        ),
        "mowa_layerwise_bridge_not_probe_source": (
            MOWA_STARFLOW_CONDITION_PROBE_FEATURE_SOURCE
            not in (forward.get("mowa_layerwise_bridge_active_heads") or [])
        ),
    }
    return {
        "stage": "P0",
        "experiment_id": "E-001",
        "training_started": False,
        "checks": checks,
        "reports": {
            "dry_run_config": str(DRY_RUN_CONFIG),
            "dry_run_report": str(DRY_RUN_REPORT),
        },
        "observed": {
            "run_id": report.get("run_id"),
            "framework": framework,
            "data": data,
            "forward": forward,
            "trainer": report.get("trainer"),
        },
        "unresolved_items": [
            "This validates StarFlowVLA + RoboCasa one-batch no-backward forward smoke only.",
            "Full training throughput, step time, and steady-state VRAM are still not measured.",
            "VRAM fields use torch CUDA allocator state in the pre-prepare dry-run path.",
            "MoWA bridge tokens are coupled only in the explicit MoWA gated dry-run config.",
            "MoWA bridge feature source must be future feature heads before launch, not starflow_condition_probe.",
            "No checkpoint/save/resume launch policy is confirmed.",
        ],
        "go_no_go": (
            "TBD: StarFlow full-path dry-run smoke passed; MoWA bridge coupling remains gated"
            if all(checks.values())
            else "No-Go: StarFlow full-path dry-run smoke incomplete"
        ),
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _is_positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and value > 0


def _starflow_ft_variant_matches_tokens(framework: dict[str, Any]) -> bool:
    variant = framework.get("starflow_ft_variant") or "config_defined"
    num_tokens = framework.get("num_target_vision_tokens")
    if not isinstance(num_tokens, int):
        return False
    if variant == "ft0":
        return num_tokens == 0
    return variant in {"config_defined", "custom", "ft_custom"} and num_tokens >= 0


if __name__ == "__main__":
    main()
