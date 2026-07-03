"""MoWA E-006 checkpoint eval-load smoke."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any


E006_CONFIG = Path("configs/mowa/mowa_e006_eval_load_smoke.yaml")
DEFAULT_CHECKPOINT = Path(
    "playground/mowa_ckpt/MoWA-E-001_starflow_ft0_save_resume_smoke_20260703_233947/checkpoints/steps_2"
)
DEFAULT_FINAL_MODEL = Path(
    "playground/mowa_ckpt/MoWA-E-001_starflow_ft0_save_resume_smoke_20260703_233947/final_model"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoWA E-006 checkpoint eval-load smoke.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--final-model", type=Path, default=DEFAULT_FINAL_MODEL)
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
    checkpoint: Path = DEFAULT_CHECKPOINT,
    final_model: Path = DEFAULT_FINAL_MODEL,
    execute_load: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root)
    checkpoint_path = root / checkpoint
    final_model_path = root / final_model
    trainer_state = _read_json(checkpoint_path / "trainer_state.json") or {}
    mapping = _read_json(checkpoint_path / "starflow_mapping.json") or {}
    load_result = _execute_model_load(checkpoint_path) if execute_load else {"executed": False}
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
        "trainer_state_step_2": trainer_state.get("completed_steps") == 2,
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
            "trainer_state": trainer_state,
            "starflow_mapping": mapping,
            "model_load": load_result,
        },
        "unresolved_items": [
            "This validates checkpoint eval-load readiness only; no policy rollout is started.",
            "E-006 baseline/zero/shuffle/head_mask policy metrics remain pending.",
            "Bridge feature source is still starflow_condition_probe, not final WAM output.",
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


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
