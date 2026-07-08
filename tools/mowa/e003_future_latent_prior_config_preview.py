"""Preview MoWA E-003 future latent prior config before dry-run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from starVLA.dataloader.mowa import DATA_GATE


CONFIG = Path("configs/mowa/mowa_e003_future_latent_prior_candidate.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoWA E-003 config preview.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--config-yaml", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_e003_future_latent_prior_config_preview(args.repo_root, args.config_yaml)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def build_e003_future_latent_prior_config_preview(
    repo_root: Path | str,
    config_yaml: Path | str = CONFIG,
) -> dict[str, Any]:
    root = Path(repo_root)
    config_path = root / Path(config_yaml)
    cfg = _load_yaml(config_path)
    latent_cache = _select(cfg, "latent_cache") or {}
    interface = _select(cfg, "interface") or {}
    status = _select(cfg, "status") or {}
    resolved_final_values = {
        "project_short_name": _select(cfg, "project_short_name"),
        "stage": _select(cfg, "stage"),
        "experiment_id": _select(cfg, "experiment_id"),
        "config_role": _select(cfg, "config_role"),
        "launch_ready": _select(cfg, "launch_guard.launch_ready"),
        "requires_human_confirmation": _select(cfg, "launch_guard.requires_human_confirmation"),
        "human_confirmed": _select(cfg, "launch_guard.human_confirmed"),
        "policy_confirmed": _select(cfg, "launch_guard.policy_confirmed"),
        "dataset_path": latent_cache.get("dataset_path"),
        "cache_root": latent_cache.get("cache_root"),
        "encoder_name": latent_cache.get("encoder_name"),
        "encoder_version": latent_cache.get("encoder_version"),
        "latent_dim": latent_cache.get("latent_dim"),
        "episode_indices": tuple(latent_cache.get("episode_indices") or ()),
        "video_keys": tuple(latent_cache.get("video_keys") or ()),
        "current_window_steps": latent_cache.get("current_window_steps"),
        "future_window_steps": latent_cache.get("future_window_steps"),
        "history_window_steps": latent_cache.get("history_window_steps"),
        "input_policy": interface.get("input_policy"),
        "target_policy": interface.get("target_policy"),
        "batch_size_smoke": interface.get("batch_size_smoke"),
        "history_latent_status": status.get("history_latent_status", DATA_GATE),
        "latent_shape_status": status.get("latent_shape_status", DATA_GATE),
        "encoder_status": status.get("encoder_status", DATA_GATE),
        "cache_artifact_status": status.get("cache_artifact_status", DATA_GATE),
    }
    checks = {
        "config_created": config_path.is_file(),
        "builder_design_config_referenced": bool(latent_cache.get("dataset_path"))
        and bool(_select(cfg, "references.builder_design_config")),
        "interface_config_referenced": bool(_select(cfg, "references.interface_config")),
        "launch_guard_closed": _select(cfg, "launch_guard.launch_ready") is False,
        "uses_fake_encoder": latent_cache.get("encoder_name") == "wan-fake-encoder"
        and latent_cache.get("encoder_version") == "fake-v1",
        "current_latent_only_input_policy": interface.get("input_policy")
        == "current_latent_plus_text_only",
        "future_latent_target_only_policy": interface.get("target_policy") == "future_latent_only",
        "history_latent_status_data_gate": status.get("history_latent_status", DATA_GATE) == DATA_GATE,
        "latent_shape_status_data_gate": status.get("latent_shape_status", DATA_GATE) == DATA_GATE,
        "encoder_status_data_gate": status.get("encoder_status", DATA_GATE) == DATA_GATE,
        "cache_artifact_status_data_gate": status.get("cache_artifact_status", DATA_GATE) == DATA_GATE,
    }
    return {
        "stage": "future_latent_prior",
        "task_id": "M3-002",
        "experiment_id": "E-003",
        "config_role": "future_latent_prior_candidate",
        "launch_ready": bool(_select(cfg, "launch_guard.launch_ready")),
        "config_path": str(Path(config_yaml)),
        "resolved_final_values": resolved_final_values,
        "checks": checks,
        "unresolved_items": [
            "This preview validates E-003 config shape and input policy only.",
            "Real Wan latent cache builder, cache parity and training launch are still gated.",
        ],
        "go_no_go": (
            "TBD: E-003 config preview passed; cache dry-run evidence still required"
            if all(checks.values())
            else "No-Go: E-003 config preview incomplete"
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
