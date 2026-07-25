"""Validate MoWA E-003 history sampling consistency across train/inference configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


E001_RUNTIME_POLICY = Path("configs/mowa/mowa_e001_runtime_policy_draft.yaml")
E001_LAUNCH_DRAFT = Path("configs/mowa/mowa_e001_launch_draft.yaml")
E001_TRAINING_COMMAND_DRAFT = Path("configs/mowa/mowa_e001_training_command_draft.yaml")
E001_READINESS_REPORT = Path("docs_zh/mowa/mowa_e001_readiness_smoke.json")
PRODUCTION_WINDOW_PREFLIGHT_REPORT = Path(
    "docs_zh/mowa/g0_atomic_core_smoke/mowa_g0_atomic_core_production_window_5hz_preflight_smoke.json"
)
TEMPORAL_PROFILE_REPORT = Path(
    "docs_zh/mowa/g0_atomic_core_smoke/mowa_g0_atomic_core_temporal_profile.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoWA E-003 history sampling consistency smoke.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_e003_history_sampling_consistency_smoke(args.repo_root)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def build_e003_history_sampling_consistency_smoke(repo_root: Path | str) -> dict[str, Any]:
    root = Path(repo_root)
    runtime_policy = _load_yaml(root / E001_RUNTIME_POLICY)
    launch_draft = _load_yaml(root / E001_LAUNCH_DRAFT)
    training_command = _load_yaml(root / E001_TRAINING_COMMAND_DRAFT)
    readiness = _read_json(root / E001_READINESS_REPORT) or {}
    production_window = _read_json(root / PRODUCTION_WINDOW_PREFLIGHT_REPORT) or {}
    temporal_profile = _read_json(root / TEMPORAL_PROFILE_REPORT) or {}

    launch_temporal = _build_temporal_bundle(
        raw_action_hz=_select(launch_draft, "training.raw_action_hz"),
        production_wam_hz=_select(launch_draft, "training.wam_hz"),
        wam_stride=_select(launch_draft, "training.wam_stride"),
        history_steps=_select(launch_draft, "training.history_steps"),
        future_steps=_select(launch_draft, "training.future_steps"),
        action_chunk_steps=_select(launch_draft, "training.action_chunk_steps"),
    )
    command_temporal = _build_temporal_bundle(
        raw_action_hz=_select(training_command, "runtime_targets.raw_action_hz"),
        production_wam_hz=_select(training_command, "runtime_targets.production_wam_hz"),
        wam_stride=_select(training_command, "runtime_targets.wam_stride"),
        history_steps=_select(training_command, "runtime_targets.history_steps"),
        future_steps=_select(training_command, "runtime_targets.future_steps"),
        action_chunk_steps=_select(training_command, "runtime_targets.action_chunk_steps"),
    )
    policy_temporal = _build_temporal_bundle(
        raw_action_hz=_select(runtime_policy, "temporal_policy.raw_action_hz"),
        production_wam_hz=_select(runtime_policy, "temporal_policy.production_wam_hz"),
        wam_stride=_select(runtime_policy, "temporal_policy.wam_stride"),
        history_steps=_select(runtime_policy, "temporal_policy.history_steps"),
        future_steps=_select(runtime_policy, "temporal_policy.future_steps"),
        action_chunk_steps=_select(runtime_policy, "temporal_policy.action_chunk_steps"),
    )
    policy_core_temporal = _build_temporal_bundle(
        raw_action_hz=_select(runtime_policy, "temporal_policy.raw_action_hz"),
        production_wam_hz=_select(runtime_policy, "temporal_policy.production_wam_hz"),
        wam_stride=None,
        history_steps=_select(runtime_policy, "temporal_policy.history_steps"),
        future_steps=_select(runtime_policy, "temporal_policy.future_steps"),
        action_chunk_steps=_select(runtime_policy, "temporal_policy.action_chunk_steps"),
    )
    readiness_temporal = _build_temporal_bundle(
        raw_action_hz=readiness.get("observed", {}).get("raw_action_hz"),
        production_wam_hz=readiness.get("observed", {}).get("production_wam_hz"),
        wam_stride=readiness.get("observed", {}).get("wam_stride"),
        history_steps=readiness.get("observed", {}).get("history_steps"),
        future_steps=readiness.get("observed", {}).get("future_steps"),
        action_chunk_steps=readiness.get("observed", {}).get("action_chunk_steps"),
    )
    production_window_temporal = _build_temporal_bundle(
        raw_action_hz=None,
        production_wam_hz=None,
        wam_stride=None,
        history_steps=(production_window.get("window_config") or {}).get("history_steps"),
        future_steps=(production_window.get("window_config") or {}).get("future_steps"),
        action_chunk_steps=(production_window.get("window_config") or {}).get("action_chunk_steps"),
    )
    readiness_window_core_temporal = _build_temporal_bundle(
        raw_action_hz=None,
        production_wam_hz=None,
        wam_stride=None,
        history_steps=readiness_temporal["history_steps"],
        future_steps=readiness_temporal["future_steps"],
        action_chunk_steps=readiness_temporal["action_chunk_steps"],
    )
    command_core_temporal = _build_temporal_bundle(
        raw_action_hz=command_temporal["raw_action_hz"],
        production_wam_hz=command_temporal["production_wam_hz"],
        wam_stride=None,
        history_steps=command_temporal["history_steps"],
        future_steps=command_temporal["future_steps"],
        action_chunk_steps=command_temporal["action_chunk_steps"],
    )
    temporal_profile_status = (
        temporal_profile.get("obs_fps_status") == "Data Gate"
        and temporal_profile.get("action_hz_status") == "Data Gate"
        and temporal_profile.get("history_window_status") == "Data Gate"
        and temporal_profile.get("future_window_status") == "Data Gate"
    )

    checks = {
        "runtime_policy_created": (root / E001_RUNTIME_POLICY).is_file(),
        "launch_draft_created": (root / E001_LAUNCH_DRAFT).is_file(),
        "training_command_draft_created": (root / E001_TRAINING_COMMAND_DRAFT).is_file(),
        "readiness_report_created": (root / E001_READINESS_REPORT).is_file(),
        "production_window_preflight_report_created": (root / PRODUCTION_WINDOW_PREFLIGHT_REPORT).is_file(),
        "temporal_profile_report_created": (root / TEMPORAL_PROFILE_REPORT).is_file(),
        "launch_draft_matches_runtime_policy": launch_temporal == policy_temporal,
        "training_command_matches_runtime_policy_core": command_core_temporal == policy_core_temporal,
        "readiness_matches_runtime_policy": readiness_temporal == policy_temporal,
        "readiness_matches_production_window_preflight_core": readiness_window_core_temporal
        == production_window_temporal,
        "launch_matches_production_window_preflight_core": _build_temporal_bundle(
            raw_action_hz=None,
            production_wam_hz=None,
            wam_stride=None,
            history_steps=launch_temporal["history_steps"],
            future_steps=launch_temporal["future_steps"],
            action_chunk_steps=launch_temporal["action_chunk_steps"],
        )
        == production_window_temporal,
        "temporal_profile_reports_data_gate_status": temporal_profile_status,
    }

    return {
        "stage": "future_latent_prior",
        "experiment_id": "E-003",
        "training_started": False,
        "checks": checks,
        "configs": {
            "runtime_policy": str(E001_RUNTIME_POLICY),
            "launch_draft": str(E001_LAUNCH_DRAFT),
            "training_command_draft": str(E001_TRAINING_COMMAND_DRAFT),
            "readiness_report": str(E001_READINESS_REPORT),
            "production_window_preflight_report": str(PRODUCTION_WINDOW_PREFLIGHT_REPORT),
            "temporal_profile_report": str(TEMPORAL_PROFILE_REPORT),
        },
        "observed": {
            "launch_temporal": launch_temporal,
            "command_temporal": command_temporal,
            "runtime_policy_temporal": policy_temporal,
            "readiness_temporal": readiness_temporal,
            "production_window_temporal": production_window_temporal,
            "temporal_profile_status": {
                "obs_fps_status": temporal_profile.get("obs_fps_status"),
                "action_hz_status": temporal_profile.get("action_hz_status"),
                "history_window_status": temporal_profile.get("history_window_status"),
                "future_window_status": temporal_profile.get("future_window_status"),
            },
        },
        "unresolved_items": [
            "This smoke only validates temporal-policy consistency; it does not build latent caches.",
            "Real Wan encoder/VAE and latent cache writer remain separately gated.",
        ],
        "go_no_go": (
            "TBD: history sampling consistency smoke passed; latent-cache builder remains gated"
            if all(checks.values())
            else "No-Go: history sampling consistency smoke incomplete"
        ),
    }


def _build_temporal_bundle(
    *,
    raw_action_hz: Any,
    production_wam_hz: Any,
    wam_stride: Any,
    history_steps: Any,
    future_steps: Any,
    action_chunk_steps: Any,
) -> dict[str, Any]:
    return {
        "raw_action_hz": raw_action_hz,
        "production_wam_hz": production_wam_hz,
        "wam_stride": wam_stride,
        "history_steps": history_steps,
        "future_steps": future_steps,
        "action_chunk_steps": action_chunk_steps,
    }


def _load_yaml(path: Path) -> Any | None:
    if not path.is_file():
        return None
    return OmegaConf.load(path)


def _select(cfg: Any | None, dot_path: str) -> Any:
    if cfg is None:
        return None
    value = OmegaConf.select(cfg, dot_path, default=None)
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
