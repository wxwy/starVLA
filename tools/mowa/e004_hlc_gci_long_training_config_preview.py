"""Preview MoWA E-004 formal long-training config before launch wiring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

DEFAULT_LONG_TRAINING_CONFIG = Path("configs/mowa/mowa_e004_hlc_gci_long_training_candidate.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoWA E-004 long-training config preview.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--config-yaml", type=Path, default=DEFAULT_LONG_TRAINING_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_e004_hlc_gci_long_training_config_preview(args.repo_root, args.config_yaml)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def build_e004_hlc_gci_long_training_config_preview(
    repo_root: Path | str,
    config_yaml: Path | str = DEFAULT_LONG_TRAINING_CONFIG,
) -> dict[str, Any]:
    root = Path(repo_root)
    config_path = root / Path(config_yaml)
    cfg = _load_yaml(config_path)
    interface = _select(cfg, "interface") or {}
    training = _select(cfg, "training") or {}
    history_contract = _select(cfg, "history_contract") or {}
    per_device_batch_size = int(training.get("per_device_batch_size") or 0)
    gradient_accumulation_steps = int(training.get("gradient_accumulation_steps") or 0)
    effective_batch_size = per_device_batch_size * gradient_accumulation_steps
    resolved_final_values = {
        "run_root_dir": _select(cfg, "run_root_dir"),
        "run_id": _select(cfg, "run_id"),
        "experiment_role": _select(cfg, "experiment_role"),
        "wandb_project": _select(cfg, "wandb_project"),
        "wandb_entity": _select(cfg, "wandb_entity"),
        "wandb_mode": _select(cfg, "wandb_mode"),
        "history_latent_dim": interface.get("history_latent_dim"),
        "condition_hidden_dim": interface.get("condition_hidden_dim"),
        "history_steps": interface.get("history_steps"),
        "compressed_history_dim": interface.get("compressed_history_dim"),
        "gate_hidden_dim": interface.get("gate_hidden_dim"),
        "history_source": history_contract.get("history_source"),
        "future_action_policy": history_contract.get("future_action_policy"),
        "per_device_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "max_train_steps": training.get("max_train_steps"),
        "num_warmup_steps": training.get("num_warmup_steps"),
        "save_interval": training.get("save_interval"),
        "eval_interval": training.get("eval_interval"),
        "checkpoint_format": training.get("checkpoint_format"),
    }
    checks = {
        "config_created": config_path.is_file(),
        "launch_guard_open": _select(cfg, "launch_guard.launch_ready") is True,
        "launch_guard_human_confirmed": _select(cfg, "launch_guard.human_confirmed") is True,
        "launch_guard_policy_confirmed": _select(cfg, "launch_guard.policy_confirmed") is True,
        "uses_4090_batch_profile": per_device_batch_size == 2
        and gradient_accumulation_steps == 16
        and effective_batch_size == 32,
        "history_source_robot_only": history_contract.get("history_source") == "robot_history_latent_only",
        "future_action_policy_target_only": history_contract.get("future_action_policy") == "target_only",
        "checkpoint_format_lightweight": training.get("checkpoint_format") == "lightweight",
        "resume_policy_latest_complete_only": training.get("resume_policy") == "resume_latest_complete_only",
    }
    return {
        "stage": "hlc_gci",
        "experiment_id": "E-004",
        "config_role": "hlcgci_long_training_candidate",
        "launch_ready": bool(_select(cfg, "launch_guard.launch_ready")),
        "config_path": str(Path(config_yaml)),
        "resolved_final_values": resolved_final_values,
        "checks": checks,
        "unresolved_items": [
            "This preview only records the formal long-training draft.",
            "The actual long-running E-004 training entrypoint is still not wired.",
            "Real shuffled-robot rollout still needs a checkpoint-backed policy path and metrics.",
        ],
        "go_no_go": (
            "TBD: long-training config preview passed; launch wiring remains pending"
            if all(checks.values())
            else "No-Go: long-training config preview incomplete"
        ),
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
