"""MoWA E-006 checkpoint eval-load smoke."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

try:
    from tools.mowa.mowa_checkpoint_resolver import (
        resolve_mowa_checkpoint_reference,
        resolve_mowa_final_model_reference,
    )
except ModuleNotFoundError:
    from mowa_checkpoint_resolver import (
        resolve_mowa_checkpoint_reference,
        resolve_mowa_final_model_reference,
    )


E006_CONFIG = Path("configs/mowa/mowa_e006_eval_load_smoke.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoWA E-006 checkpoint eval-load smoke.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--final-model", type=Path, default=None)
    parser.add_argument("--execute-load", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_e006_eval_load_smoke(
        args.repo_root,
        checkpoint=args.checkpoint,
        final_model=args.final_model,
        execute_load=args.execute_load,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def build_e006_eval_load_smoke(
    repo_root: Path | str,
    *,
    checkpoint: Path | None = None,
    final_model: Path | None = None,
    execute_load: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root)
    if checkpoint is None or final_model is None:
        default_checkpoint, default_final_model = _load_default_checkpoint_paths(root)
        checkpoint = checkpoint or default_checkpoint
        final_model = final_model or default_final_model
    checkpoint_path = root / checkpoint
    final_model_path = root / final_model
    trainer_state = _read_json(checkpoint_path / "trainer_state.json") or {}
    mapping = _read_json(checkpoint_path / "starflow_mapping.json") or {}
    load_result = _execute_model_load(checkpoint_path) if execute_load else {"executed": False}
    expected_completed_steps = _expected_completed_steps_from_checkpoint(checkpoint)
    checks = {
        "config_created": (root / E006_CONFIG).is_file(),
        "checkpoint_under_mowa_ckpt": str(checkpoint).startswith("playground/mowa_ckpt/"),
        "checkpoint_dir_exists": checkpoint_path.is_dir(),
        "final_model_dir_exists": final_model_path.is_dir(),
        "model_index_exists": (checkpoint_path / "model.safetensors.index.json").is_file(),
        "model_shards_exist": len(list(checkpoint_path.glob("model-*.safetensors"))) > 0,
        "config_full_exists": (checkpoint_path / "config.full.yaml").is_file(),
        "dataset_statistics_exists": (checkpoint_path / "dataset_statistics.json").is_file(),
        "starflow_mapping_exists": (checkpoint_path / "starflow_mapping.json").is_file(),
        "trainer_state_matches_checkpoint": (
            expected_completed_steps is not None
            and trainer_state.get("completed_steps") == expected_completed_steps
        ),
        "optimizer_state_exists": (checkpoint_path / "optimizer_rank_00000.pt").is_file(),
        "scheduler_state_exists": (checkpoint_path / "scheduler.pt").is_file(),
        "rng_state_exists": any(checkpoint_path.glob("random_states_*.pkl")),
        "mapping_framework_starflow": mapping.get("framework_name") == "StarFlowVLA",
        "mapping_action_head_layerwisefm": mapping.get("action_head") == "LayerwiseFM",
        "model_load_executed_or_not_required": (
            load_result.get("executed") is False or load_result.get("loaded") is True
        ),
    }
    return {
        "stage": "P0",
        "experiment_id": "E-006",
        "eval_started": False,
        "policy_eval_started": False,
        "execute_load": execute_load,
        "checks": checks,
        "configs": {
            "eval_load_smoke_config": str(E006_CONFIG),
        },
        "observed": {
            "checkpoint": str(checkpoint),
            "final_model": str(final_model),
            "expected_completed_steps": expected_completed_steps,
            "trainer_state": trainer_state,
            "starflow_mapping": mapping,
            "model_load": load_result,
        },
        "unresolved_items": [
            "This validates checkpoint eval-load readiness only; no policy rollout is started.",
            "E-006 baseline/zero/shuffle/head_mask policy metrics remain pending.",
            "Checkpoint uses mowa_p0_fullheads bridge feature source, but action-gain evidence is still pending.",
        ],
        "go_no_go": (
            "TBD: E-006 eval-load smoke passed; policy rollout remains gated"
            if all(checks.values())
            else "No-Go: E-006 eval-load smoke incomplete"
        ),
    }


def _execute_model_load(checkpoint_path: Path) -> dict[str, Any]:
    start_time = time.perf_counter()
    try:
        import torch
        from omegaconf import OmegaConf

        from starVLA.model.framework.base_framework import build_framework
        from starVLA.model.framework.share_tools import apply_config_compat
        from starVLA.training.trainer_utils.trainer_tools import TrainerUtils

        cfg = OmegaConf.load(checkpoint_path / "config.full.yaml")
        cfg = apply_config_compat(cfg)
        model = build_framework(cfg)
        TrainerUtils.load_pretrained_backbones(
            model,
            str(checkpoint_path),
            preferred_format="safetensors",
        )
        model.eval()
        param_count = sum(param.numel() for param in model.parameters())
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "executed": True,
            "loaded": True,
            "elapsed_sec": time.perf_counter() - start_time,
            "param_count": int(param_count),
        }
    except Exception as exc:
        return {
            "executed": True,
            "loaded": False,
            "elapsed_sec": time.perf_counter() - start_time,
            "error": repr(exc),
        }


def _load_default_checkpoint_paths(repo_root: Path) -> tuple[Path, Path]:
    from omegaconf import OmegaConf

    config_path = repo_root / E006_CONFIG
    cfg = OmegaConf.load(config_path)
    checkpoint_cfg = cfg.checkpoint
    checkpoint_root_policy = Path(
        str(getattr(checkpoint_cfg, "checkpoint_root_policy", "playground/mowa_ckpt"))
    )
    return (
        resolve_mowa_checkpoint_reference(
            repo_root,
            checkpoint_cfg.eval_candidate_checkpoint,
            checkpoint_root_policy=checkpoint_root_policy,
        ),
        resolve_mowa_final_model_reference(
            repo_root,
            checkpoint_cfg.final_model_checkpoint,
            checkpoint_root_policy=checkpoint_root_policy,
        ),
    )


def _expected_completed_steps_from_checkpoint(checkpoint: Path) -> int | None:
    checkpoint_name = Path(checkpoint).name
    if not checkpoint_name.startswith("steps_"):
        return None
    suffix = checkpoint_name.removeprefix("steps_")
    return int(suffix) if suffix.isdigit() else None


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
