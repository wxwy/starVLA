"""Validate StarFlow ft0 baseline vs MoWA bridge candidate wiring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
    "trainer.disable_wandb",
    "trainer.save_interval",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoWA E-001 StarFlow ft0 comparison smoke.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_starflow_ft0_comparison_smoke(args.repo_root)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def build_starflow_ft0_comparison_smoke(repo_root: Path | str) -> dict[str, Any]:
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
    checks = {
        "baseline_config_created": (root / BASELINE_CONFIG).is_file(),
        "mowa_candidate_config_created": (root / MOWA_CONFIG).is_file(),
        "official_starflow_ft0_config_exists": (root / OFFICIAL_FT0_CONFIG).is_file(),
        "baseline_launch_gated": _select(baseline, "launch_guard.launch_ready") is False,
        "mowa_launch_gated": _select(mowa, "launch_guard.launch_ready") is False,
        "baseline_bridge_disabled": (
            _select(baseline, "framework.mowa.enable_layerwise_bridge_token_coupling") is False
        ),
        "mowa_bridge_enabled": (
            _select(mowa, "framework.mowa.enable_layerwise_bridge_token_coupling") is True
        ),
        "baseline_p0_labels_disabled": (
            _select(baseline, "datasets.vla_data.enable_mowa_p0_labels") is False
        ),
        "mowa_p0_labels_enabled": (
            _select(mowa, "datasets.vla_data.enable_mowa_p0_labels") is True
        ),
        "paired_invariants_match": all(invariant_matches.values()),
        "official_ft0_is_reference_only": (
            _select(official, "framework.name") == "StarFlowVLA"
            and _select(official, "framework.action_model.num_target_vision_tokens") == 0
            and _select(official, "datasets.vla_data.data_mix")
            != _select(mowa, "datasets.vla_data.data_mix")
        ),
    }
    return {
        "stage": "P0",
        "experiment_id": "E-001",
        "training_started": False,
        "baseline_config": str(BASELINE_CONFIG),
        "mowa_config": str(MOWA_CONFIG),
        "official_starflow_ft0_reference": str(OFFICIAL_FT0_CONFIG),
        "checks": checks,
        "invariant_values": invariant_values,
        "invariant_matches": invariant_matches,
        "expected_differences": {
            "framework.mowa.enable_layerwise_bridge_token_coupling": {
                "baseline": False,
                "mowa": True,
            },
            "datasets.vla_data.enable_mowa_p0_labels": {
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
            "Both paired candidate configs remain launch-gated until explicit human confirmation.",
        ],
        "go_no_go": (
            "TBD: paired StarFlow ft0 baseline and MoWA bridge configs are aligned and gated"
            if all(checks.values())
            else "No-Go: paired StarFlow ft0 comparison config drift detected"
        ),
    }


def _select(cfg: Any, dot_path: str) -> Any:
    value = OmegaConf.select(cfg, dot_path, default=None)
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


if __name__ == "__main__":
    main()
