"""Validate StarFlow ft0 baseline vs MoWA bridge candidate wiring."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from omegaconf import OmegaConf


BASELINE_CONFIG = Path("configs/mowa/mowa_e001_starflow_ft0_baseline_candidate.yaml")
MOWA_CONFIG = Path("configs/mowa/mowa_e001_starflow_ft0_launch_candidate.yaml")
OFFICIAL_FT0_CONFIG = Path("configs/starflow_vla/ablations/future_tokens_0.yaml")
OUTPUT = Path("docs_zh/mowa/mowa_e001_starflow_ft0_comparison_smoke.json")

INVARIANT_PATHS = (
    "framework.name",
    "framework.starflow_ft_variant",
    "framework.state_mode",
    "framework.qwenvl.base_vlm",
    "framework.qwenvl.attn_implementation",
    "framework.qwenvl.vl_hidden_dim",
    "framework.qwenvl.num_vl_layers",
    "framework.action_model.action_model_type",
    "framework.action_model.action_dim",
    "framework.action_model.state_dim",
    "framework.action_model.action_horizon",
    "framework.action_model.future_action_window_size",
    "framework.action_model.past_action_window_size",
    "framework.action_model.repeated_diffusion_steps",
    "framework.action_model.num_inference_timesteps",
    "framework.action_model.add_pos_embed",
    "framework.action_model.max_seq_len",
    "framework.action_model.num_target_vision_tokens",
    "framework.action_model.diffusion_model_cfg.action_dit_hidden_dim",
    "datasets.vla_data.data_mix",
    "datasets.vla_data.action_type",
    "datasets.vla_data.per_device_batch_size",
    "trainer.max_train_steps",
    "trainer.gradient_accumulation_steps",
    "wandb_mode",
    "trainer.save_interval",
)
ALLOWED_RUNTIME_DIFFERENCE_PATHS = (
    "run_id",
    "launch_guard.reason",
    "launch_guard.launch_ready",
    "launch_guard.policy_confirmed",
    "launch_guard.human_confirmed",
    "framework.mowa.enable_layerwise_bridge_token_coupling",
    "framework.mowa.enable_future_supervision_loss",
    "framework.mowa.layerwise_bridge_feature_source",
    "framework.mowa.layerwise_bridge_active_heads",
    "framework.mowa.wam_feature_dim",
    "framework.mowa.action_hidden_dim",
    "framework.mowa.num_bridge_tokens",
    "datasets.vla_data.enable_mowa_future_labels",
    "trainer.enable_mowa_future_supervision_loss",
)
REQUIRED_RUNTIME_DIFFERENCE_PATHS = (
    "framework.mowa.enable_layerwise_bridge_token_coupling",
    "framework.mowa.enable_future_supervision_loss",
    "datasets.vla_data.enable_mowa_future_labels",
    "trainer.enable_mowa_future_supervision_loss",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoWA E-001 StarFlow ft0 comparison smoke.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--check-launch-guard",
        action="store_true",
        help="Execute train_starvla.py up to launch_guard for both candidate configs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_starflow_ft0_comparison_smoke(
        args.repo_root,
        check_launch_guard=args.check_launch_guard,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


CommandRunner = Callable[[list[str], Path], Any]


def build_starflow_ft0_comparison_smoke(
    repo_root: Path | str,
    *,
    check_launch_guard: bool = False,
    command_runner: CommandRunner | None = None,
) -> dict[str, Any]:
    root = Path(repo_root)
    baseline = OmegaConf.load(root / BASELINE_CONFIG)
    mowa = OmegaConf.load(root / MOWA_CONFIG)
    official = OmegaConf.load(root / OFFICIAL_FT0_CONFIG)
    invariant_values = {
        path: {
            "baseline": _select(baseline, path),
            "mowa": _select(mowa, path),
        }
        for path in INVARIANT_PATHS
    }
    invariant_matches = {
        path: values["baseline"] == values["mowa"]
        for path, values in invariant_values.items()
    }
    runtime_symmetry = _build_runtime_symmetry_report(baseline, mowa)
    checks = {
        "baseline_config_created": (root / BASELINE_CONFIG).is_file(),
        "mowa_candidate_config_created": (root / MOWA_CONFIG).is_file(),
        "official_starflow_ft0_config_exists": (root / OFFICIAL_FT0_CONFIG).is_file(),
        "baseline_launch_gated": _select(baseline, "launch_guard.launch_ready") is False,
        "mowa_launch_state_consistent": _launch_guard_state_is_consistent(mowa),
        "baseline_bridge_disabled": (
            _select(baseline, "framework.mowa.enable_layerwise_bridge_token_coupling") is False
        ),
        "mowa_bridge_enabled": (
            _select(mowa, "framework.mowa.enable_layerwise_bridge_token_coupling") is True
        ),
        "baseline_future_labels_disabled": (
            _select(baseline, "datasets.vla_data.enable_mowa_future_labels") is False
        ),
        "mowa_future_labels_enabled": (
            _select(mowa, "datasets.vla_data.enable_mowa_future_labels") is True
        ),
        "paired_invariants_match": all(invariant_matches.values()),
        "paired_runtime_symmetry_passed": runtime_symmetry["passed"],
        "official_ft0_is_reference_only": (
            _select(official, "framework.name") == "StarFlowVLA"
            and _select(official, "framework.action_model.num_target_vision_tokens") == 0
            and _select(official, "datasets.vla_data.data_mix")
            != _select(mowa, "datasets.vla_data.data_mix")
        ),
    }
    launch_guard_execution = {
        "checked": check_launch_guard,
        "baseline": None,
        "mowa": None,
    }
    if check_launch_guard:
        launch_guard_execution["baseline"] = _run_launch_guard_check(
            root,
            BASELINE_CONFIG,
            command_runner=command_runner,
        )
        launch_guard_execution["mowa"] = _run_launch_guard_check(
            root,
            MOWA_CONFIG,
            command_runner=command_runner,
        )
        checks["baseline_launch_guard_blocks_entrypoint"] = bool(
            launch_guard_execution["baseline"]["blocked_by_launch_guard"]
        )
        checks["mowa_launch_guard_blocks_entrypoint"] = bool(
            launch_guard_execution["mowa"]["blocked_by_launch_guard"]
        )
    return {
        "stage": "full_heads",
        "experiment_id": "E-001",
        "training_started": False,
        "baseline_config": str(BASELINE_CONFIG),
        "mowa_config": str(MOWA_CONFIG),
        "official_starflow_ft0_reference": str(OFFICIAL_FT0_CONFIG),
        "checks": checks,
        "launch_guard_execution": launch_guard_execution,
        "invariant_values": invariant_values,
        "invariant_matches": invariant_matches,
        "runtime_symmetry": runtime_symmetry,
        "expected_differences": {
            "framework.mowa.enable_layerwise_bridge_token_coupling": {
                "baseline": False,
                "mowa": True,
            },
            "datasets.vla_data.enable_mowa_future_labels": {
                "baseline": False,
                "mowa": True,
            },
            "framework.mowa.layerwise_bridge_feature_source": {
                "baseline": None,
                "mowa": _select(mowa, "framework.mowa.layerwise_bridge_feature_source"),
            },
            "framework.mowa.num_bridge_tokens": {
                "baseline": None,
                "mowa": _select(mowa, "framework.mowa.num_bridge_tokens"),
            },
        },
        "unresolved_items": [
            "This is a static config comparison only; no training is started.",
            "The official StarFlow ft0 YAML remains a reference config on LIBERO, not the paired RoboCasa baseline.",
            (
                "The paired baseline remains launch-gated while the MoWA candidate is launch-approved."
                if _select(mowa, "launch_guard.launch_ready") is True
                else "Both paired candidate configs remain launch-gated until explicit human confirmation."
            ),
        ],
        "go_no_go": (
            "TBD: paired StarFlow ft0 baseline and MoWA bridge configs are aligned"
            if all(checks.values())
            else "No-Go: paired StarFlow ft0 comparison config drift detected"
        ),
    }


def _select(cfg: Any, dot_path: str) -> Any:
    value = OmegaConf.select(cfg, dot_path, default=None)
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def _launch_guard_state_is_consistent(cfg: Any) -> bool:
    launch_ready = _select(cfg, "launch_guard.launch_ready")
    policy_confirmed = _select(cfg, "launch_guard.policy_confirmed")
    requires_human_confirmation = _select(cfg, "launch_guard.requires_human_confirmation")
    human_confirmed = _select(cfg, "launch_guard.human_confirmed")
    if launch_ready is False:
        return policy_confirmed is False
    if launch_ready is True:
        if policy_confirmed is not True:
            return False
        if requires_human_confirmation is True:
            return human_confirmed is True
        return True
    return False


def _build_runtime_symmetry_report(baseline: Any, mowa: Any) -> dict[str, Any]:
    baseline_flat = _flatten_mapping(OmegaConf.to_container(baseline, resolve=True))
    mowa_flat = _flatten_mapping(OmegaConf.to_container(mowa, resolve=True))
    all_paths = sorted(set(baseline_flat) | set(mowa_flat))
    differences = {
        path: {
            "baseline": baseline_flat.get(path),
            "mowa": mowa_flat.get(path),
        }
        for path in all_paths
        if baseline_flat.get(path) != mowa_flat.get(path)
    }
    allowed_difference_paths = tuple(
        path for path in differences if path in ALLOWED_RUNTIME_DIFFERENCE_PATHS
    )
    unexpected_difference_paths = tuple(
        path for path in differences if path not in ALLOWED_RUNTIME_DIFFERENCE_PATHS
    )
    missing_required_difference_paths = tuple(
        path for path in REQUIRED_RUNTIME_DIFFERENCE_PATHS if path not in differences
    )
    return {
        "passed": (
            not unexpected_difference_paths
            and not missing_required_difference_paths
        ),
        "compared_path_count": len(all_paths),
        "matched_path_count": len(all_paths) - len(differences),
        "allowed_difference_paths": allowed_difference_paths,
        "unexpected_difference_paths": unexpected_difference_paths,
        "missing_required_difference_paths": missing_required_difference_paths,
        "allowed_differences": {
            path: differences[path] for path in allowed_difference_paths
        },
        "unexpected_differences": {
            path: differences[path] for path in unexpected_difference_paths
        },
    }


def _flatten_mapping(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flattened: dict[str, Any] = {}
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_mapping(item, path))
        return flattened
    return {prefix: value}


def _run_launch_guard_check(
    repo_root: Path,
    config_path: Path,
    *,
    command_runner: CommandRunner | None = None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "starVLA/training/train_starvla.py",
        "--config_yaml",
        str(config_path),
    ]
    runner = command_runner or _default_command_runner
    completed = runner(command, repo_root)
    stdout = getattr(completed, "stdout", "") or ""
    stderr = getattr(completed, "stderr", "") or ""
    combined_output = stdout + "\n" + stderr
    blocked = (
        getattr(completed, "returncode", 0) != 0
        and "Training launch blocked by launch_guard" in combined_output
    )
    return {
        "config": str(config_path),
        "command": command,
        "returncode": getattr(completed, "returncode", None),
        "blocked_by_launch_guard": blocked,
        "stdout_tail": stdout[-1000:],
        "stderr_tail": stderr[-1000:],
    }


def _default_command_runner(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


if __name__ == "__main__":
    main()
