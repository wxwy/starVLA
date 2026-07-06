"""Preview MoWA E-001 long-training config final values before launch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

DEFAULT_LONG_TRAINING_CONFIG = Path("configs/mowa/mowa_e001_starflow_ft0_long_training_candidate.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoWA E-001 long-training config preview.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--config-yaml", type=Path, default=DEFAULT_LONG_TRAINING_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_e001_long_training_config_preview(args.repo_root, args.config_yaml)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def build_e001_long_training_config_preview(
    repo_root: Path | str,
    config_yaml: Path | str = DEFAULT_LONG_TRAINING_CONFIG,
) -> dict[str, Any]:
    root = Path(repo_root)
    config_path = root / Path(config_yaml)
    cfg = _load_yaml(config_path)
    experiment_role = _select(cfg, "experiment_role")
    per_device_batch_size = _select(cfg, "datasets.vla_data.per_device_batch_size")
    gradient_accumulation_steps = _select(cfg, "trainer.gradient_accumulation_steps")
    effective_batch_size = (per_device_batch_size or 0) * (gradient_accumulation_steps or 0)

    resolved_final_values = {
        "run_root_dir": _select(cfg, "run_root_dir"),
        "run_id": _select(cfg, "run_id"),
        "experiment_role": _select(cfg, "experiment_role"),
        "gpu_profile": "RTX_4090_single_card_target",
        "per_device_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "max_train_steps": _select(cfg, "trainer.max_train_steps"),
        "num_warmup_steps": _select(cfg, "trainer.num_warmup_steps"),
        "save_interval": _select(cfg, "trainer.save_interval"),
        "eval_interval": _select(cfg, "trainer.eval_interval"),
        "logging_frequency": _select(cfg, "trainer.logging_frequency"),
        "wandb_project": _select(cfg, "wandb_project"),
        "wandb_entity": _select(cfg, "wandb_entity"),
        "wandb_mode": _select(cfg, "wandb_mode"),
        "freeze_modules": _select(cfg, "trainer.freeze_modules"),
        "enable_future_supervision_loss": _select(cfg, "trainer.enable_mowa_future_supervision_loss"),
        "enable_layerwise_bridge_token_coupling": _select(
            cfg, "framework.mowa.enable_layerwise_bridge_token_coupling"
        ),
        "layerwise_bridge_feature_source": _select(
            cfg, "framework.mowa.layerwise_bridge_feature_source"
        ),
        "enable_mowa_future_labels": _select(cfg, "datasets.vla_data.enable_mowa_future_labels"),
        "checkpoint_format": _select(cfg, "trainer.checkpoint_format"),
        "save_checkpoint_as_directory": _select(cfg, "trainer.save_checkpoint_as_directory"),
    }
    checks = {
        "config_created": config_path.is_file(),
        "uses_4090_batch_profile": per_device_batch_size == 1
        and gradient_accumulation_steps == 32
        and effective_batch_size == 32,
        "marks_mowa_main_experiment_role": _select(cfg, "experiment_role") == "mowa_main",
        "max_steps_not_smoke_1000": _select(cfg, "trainer.max_train_steps") == 80000,
        "freeze_modules_is_qwen_vl_interface": _select(cfg, "trainer.freeze_modules")
        == "qwen_vl_interface",
        "wandb_project_is_mowa": _select(cfg, "wandb_project") == "MoWA",
        "wandb_entity_is_expected": _select(cfg, "wandb_entity")
        == "silencewx-harbin-institute-of-technology",
        "wandb_mode_is_online": _select(cfg, "wandb_mode") == "online",
        "keeps_future_supervision_chain": _select(cfg, "trainer.enable_mowa_future_supervision_loss")
        is True
        and _select(cfg, "framework.mowa.enable_future_supervision_loss") is True
        and _select(cfg, "framework.mowa.layerwise_bridge_feature_source")
        == "mowa_future_feature_heads",
        "keeps_mowa_ckpt_root": _select(cfg, "run_root_dir") == "playground/mowa_ckpt",
    }
    main_launch_ready = all(checks.values()) and experiment_role == "mowa_main"
    baseline_launch_ready = (
        checks["config_created"]
        and checks["uses_4090_batch_profile"]
        and checks["max_steps_not_smoke_1000"]
        and checks["freeze_modules_is_qwen_vl_interface"]
        and checks["wandb_project_is_mowa"]
        and checks["wandb_entity_is_expected"]
        and checks["wandb_mode_is_online"]
        and checks["keeps_mowa_ckpt_root"]
        and experiment_role == "mowa_baseline"
        and not checks["keeps_future_supervision_chain"]
    )
    return {
        "stage": "full_heads",
        "experiment_id": "E-001",
        "config_role": "long_training_launch_config",
        "launch_ready": bool(_select(cfg, "launch_guard.launch_ready")),
        "config_path": str(Path(config_yaml)),
        "resolved_final_values": resolved_final_values,
        "checks": checks,
        "go_no_go": (
            "TBD: long-training config preview passed; launch approved"
            if bool(_select(cfg, "launch_guard.launch_ready")) and (main_launch_ready or baseline_launch_ready)
            else "No-Go: long-training config preview incomplete"
        ),
        "unresolved_items": [
            "formal long-training launch command remains to be wired separately",
            "human confirmation is still required before any training launch",
        ],
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


if __name__ == "__main__":
    main()
