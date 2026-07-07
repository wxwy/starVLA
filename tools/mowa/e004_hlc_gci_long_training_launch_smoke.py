"""Run the E-004 formal long-training launch smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from omegaconf import OmegaConf

try:
    from tools.mowa.e004_hlc_gci_long_training_config_preview import (
        build_e004_hlc_gci_long_training_config_preview,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - direct script execution fallback
    if exc.name != "tools":
        raise
    from e004_hlc_gci_long_training_config_preview import (  # type: ignore[no-redef]
        build_e004_hlc_gci_long_training_config_preview,
    )

CONFIG = Path("configs/mowa/mowa_e004_hlc_gci_long_training_launch_candidate.yaml")
COMMAND_CONFIG = Path("configs/mowa/mowa_e004_training_command_long_training_candidate.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MoWA E-004 long-training launch smoke.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--config-yaml", type=Path, default=CONFIG)
    parser.add_argument("--command-config-yaml", type=Path, default=COMMAND_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_e004_hlc_gci_long_training_launch_smoke(
        args.repo_root,
        args.config_yaml,
        args.command_config_yaml,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def build_e004_hlc_gci_long_training_launch_smoke(
    repo_root: Path | str,
    config_yaml: Path | str = CONFIG,
    command_config_yaml: Path | str = COMMAND_CONFIG,
) -> dict[str, Any]:
    root = Path(repo_root)
    config_path = root / Path(config_yaml)
    command_path = root / Path(command_config_yaml)
    preview = build_e004_hlc_gci_long_training_config_preview(root, config_yaml)
    cfg = _load_yaml(config_path)
    command_cfg = _load_yaml(command_path)
    checks = {
        "long_training_launch_config_created": config_path.is_file(),
        "training_command_long_config_created": command_path.is_file(),
        "preview_passed": preview.get("go_no_go", "").startswith("TBD: long-training config preview passed"),
        "launch_guard_open": _select(cfg, "launch_guard.launch_ready") is True,
        "launch_guard_policy_confirmed": _select(cfg, "launch_guard.policy_confirmed") is True,
        "launch_guard_human_confirmed": _select(cfg, "launch_guard.human_confirmed") is True,
        "entrypoint_is_train_starvla": _select(command_cfg, "command_candidate.entrypoint")
        == "starVLA/training/train_starvla.py",
        "command_points_to_launch_config": _select(command_cfg, "command_candidate.config_yaml")
        == str(Path(config_yaml)),
        "command_mentions_train_starvla": _text_contains(
            command_path,
            ".venv/bin/python starVLA/training/train_starvla.py",
        ),
        "run_id_is_long_training": str(_select(cfg, "run_id") or "").endswith("_4090_long"),
        "wandb_mode_is_online": _select(cfg, "wandb_mode") == "online",
        "checkpoint_format_lightweight": _select(cfg, "trainer.checkpoint_format") == "lightweight",
        "resume_policy_latest_complete_only": _select(cfg, "trainer.resume_policy")
        == "resume_latest_complete_only",
        "hlcgci_integrated_in_training_framework": _tree_contains(
            root,
            ("starVLA/training", "starVLA/model/framework"),
            "MoWAHLCGCI",
        ),
        "history_latent_batch_contract_integrated_in_runtime_paths": _tree_contains(
            root,
            ("starVLA/training", "starVLA/model/framework", "starVLA/dataloader/gr00t_lerobot"),
            "history_latent",
        ),
        "hlcgci_training_objective_integrated_in_training_loop": _tree_contains_any(
            root,
            ("starVLA/training", "starVLA/model/framework"),
            ("gate_values", "compressed_history"),
        ),
    }
    return {
        "stage": "hlc_gci",
        "experiment_id": "E-004",
        "config_path": str(Path(config_yaml)),
        "command_config_path": str(Path(command_config_yaml)),
        "launch_ready": bool(_select(cfg, "launch_guard.launch_ready")),
        "checks": checks,
        "preview": preview,
        "command_candidate": _to_plain_dict(_load_yaml(command_path)),
        "go_no_go": (
            "TBD: E-004 long-training launch candidate passed; training command is wired"
            if all(checks.values())
            else "No-Go: E-004 long-training launch candidate is not training-integrated"
        ),
        "unresolved_items": [
            "Training framework wiring is present but a real-scale history latent cache is still needed.",
            "The long-training config reuses the smoke cache root (3 episodes); replace with production cache.",
            "No E-004-specific train_starvla full-path dry-run has been executed end-to-end.",
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


def _text_contains(path: Path, pattern: str) -> bool:
    if not path.is_file():
        return False
    return pattern in path.read_text(encoding="utf-8")


def _to_plain_dict(payload: Any | None) -> dict[str, Any]:
    if payload is None:
        return {}
    if OmegaConf.is_config(payload):
        return OmegaConf.to_container(payload, resolve=True)  # type: ignore[return-value]
    return dict(payload)


def _tree_contains(root: Path, relative_dirs: Iterable[str], pattern: str) -> bool:
    for relative_dir in relative_dirs:
        base = root / relative_dir
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            if pattern in path.read_text(encoding="utf-8"):
                return True
    return False


def _tree_contains_any(root: Path, relative_dirs: Iterable[str], patterns: Iterable[str]) -> bool:
    return any(_tree_contains(root, relative_dirs, pattern) for pattern in patterns)


if __name__ == "__main__":
    main()
