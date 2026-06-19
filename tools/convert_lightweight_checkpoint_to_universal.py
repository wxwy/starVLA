#!/usr/bin/env python
"""Convert a StarVLA lightweight checkpoint to a DeepSpeed Universal Checkpoint.

This tool loads a lightweight rank-sharded checkpoint (model-*.safetensors +
optimizer_rank_*.pt + scheduler/RNG state) into a DeepSpeed ZeRO engine and
rewrites it as a DeepSpeed native ZeRO checkpoint, then invokes
`deepspeed.checkpoint.ds_to_universal` to produce the Universal Checkpoint
layout that can be resumed with `load_universal: true`.

Run with the same `accelerate launch` settings used for training, e.g.:

    accelerate launch \
        --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
        --num_processes 4 \
        tools/convert_lightweight_checkpoint_to_universal.py \
        --source_checkpoint playground/Checkpoints/P0-M5-E-H2a-04_starflow_libero-goal_qwen3vl4b_lwfm_ft32_250615/checkpoints/steps_31500 \
        --output_universal_dir playground/Checkpoints/P0-M5-E-H2a-04_universal/steps_31500
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Ensure repo root is importable.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omegaconf import OmegaConf

# Importing train_starvla creates a global Accelerator; this is intentional and
# must be done under `accelerate launch` so the distributed environment is set.
from starVLA.training import train_starvla as train_script
from starVLA.training.train_starvla import (
    VLATrainer,
    accelerator,
    logger,
    prepare_data,
    setup_directories,
    setup_optimizer_and_scheduler,
)
from starVLA.training.trainer_utils.config_tracker import wrap_config
from starVLA.model.framework.base_framework import build_framework


REQUIRED_LIGHTWEIGHT_FILES = [
    "model.safetensors.index.json",
    "scheduler.pt",
    "scaler.pt",
    "trainer_state.json",
    "random_states_0.pkl",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a StarVLA lightweight checkpoint to DeepSpeed Universal Checkpoint"
    )
    parser.add_argument(
        "--source_checkpoint",
        type=str,
        required=True,
        help="Path to the lightweight checkpoint directory (e.g. .../checkpoints/steps_31500)",
    )
    parser.add_argument(
        "--output_universal_dir",
        type=str,
        required=True,
        help="Path where the Universal Checkpoint directory will be written",
    )
    parser.add_argument(
        "--temp_zero_dir",
        type=str,
        default=None,
        help="Optional temporary directory for the intermediate DeepSpeed ZeRO checkpoint",
    )
    parser.add_argument(
        "--keep_temp_zero",
        action="store_true",
        help="Keep the intermediate DeepSpeed ZeRO checkpoint after conversion",
    )
    parser.add_argument(
        "--config_yaml",
        type=str,
        default=None,
        help="Override config YAML; defaults to source_checkpoint/config.full.yaml",
    )
    parser.add_argument(
        "--ds_to_universal_timeout",
        type=int,
        default=7200,
        help="Timeout in seconds for the ds_to_universal conversion subprocess",
    )
    return parser.parse_args()


def validate_source_checkpoint(source_path: Path, expected_world_size: int = 4) -> dict:
    """Validate that the source directory looks like a complete lightweight checkpoint."""
    if not source_path.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {source_path}")
    if not source_path.is_dir():
        raise NotADirectoryError(f"Source checkpoint is not a directory: {source_path}")

    missing = [name for name in REQUIRED_LIGHTWEIGHT_FILES if not (source_path / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Source checkpoint is missing required lightweight files: {missing}"
        )

    trainer_state_path = source_path / "trainer_state.json"
    with open(trainer_state_path, "r", encoding="utf-8") as f:
        trainer_state = json.load(f)

    if trainer_state.get("checkpoint_type") != "lightweight_training":
        raise ValueError(
            f"Expected checkpoint_type='lightweight_training', got {trainer_state.get('checkpoint_type')}"
        )
    if trainer_state.get("optimizer_format") != "rank_sharded":
        raise ValueError(
            f"Expected optimizer_format='rank_sharded', got {trainer_state.get('optimizer_format')}"
        )

    saved_world_size = int(trainer_state.get("optimizer_world_size", 0))
    if saved_world_size != expected_world_size:
        raise ValueError(
            f"Lightweight checkpoint was saved with world_size={saved_world_size}, "
            f"but this conversion requires world_size={expected_world_size}"
        )

    for rank in range(saved_world_size):
        opt_file = source_path / f"optimizer_rank_{rank:05d}.pt"
        if not opt_file.exists():
            raise FileNotFoundError(f"Missing optimizer shard for rank {rank}: {opt_file}")

    # At least one model shard must exist.
    model_shards = sorted(source_path.glob("model-*.safetensors"))
    if not model_shards:
        raise FileNotFoundError(f"No model-*.safetensors shards found in {source_path}")

    return trainer_state


def build_temp_run_dir(source_path: Path, temp_root: Path, accel) -> Path:
    """Create a temporary run directory with the source checkpoint symlinked.

    Only the main process creates the directory/symlink; all other ranks wait
    until the marker file appears and then return the same path.
    """
    step_match = source_path.name
    marker_name = ".convert_run_id"
    marker_path = temp_root / marker_name

    if accel.is_main_process:
        # Use a unique directory name and write the marker atomically.
        run_id = f"convert_tmp_{int(time.time())}_{os.getpid()}"
        run_dir = temp_root / run_id
        checkpoints_dir = run_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        symlink_target = checkpoints_dir / step_match
        if symlink_target.exists() or symlink_target.is_symlink():
            symlink_target.unlink()
        symlink_target.symlink_to(source_path.resolve(), target_is_directory=True)

        # Write marker so other ranks know which directory to use.
        marker_path.write_text(run_id, encoding="utf-8")
    else:
        run_dir = None
        # Poll briefly until the marker file is written by rank 0.
        for _ in range(600):
            if marker_path.exists():
                run_id = marker_path.read_text(encoding="utf-8").strip()
                run_dir = temp_root / run_id
                if (run_dir / "checkpoints" / step_match).exists():
                    break
            time.sleep(0.1)
        if run_dir is None:
            raise RuntimeError("Timeout waiting for rank 0 to create temp run directory")

    accel.wait_for_everyone()
    return run_dir


def clear_temp_run_marker(temp_root: Path, accel) -> None:
    """Remove the marker file used to broadcast the temp run id."""
    if accel.is_main_process:
        marker_path = temp_root / ".convert_run_id"
        marker_path.unlink(missing_ok=True)


def load_source_config(source_path: Path, config_yaml: str | None) -> OmegaConf:
    """Load the source config, preferring the provided YAML or config.full.yaml."""
    if config_yaml is not None:
        cfg_path = Path(config_yaml)
    else:
        cfg_path = source_path / "config.full.yaml"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    cfg = OmegaConf.load(cfg_path)
    return cfg


def patch_config_for_conversion(cfg: OmegaConf, run_dir: Path, run_id: str) -> OmegaConf:
    """Override config so that the trainer loads the checkpoint and stops before training."""
    overrides = OmegaConf.create({
        "run_root_dir": str(run_dir),
        "run_id": run_id,
        "wandb_project": "",
        "wandb_entity": "",
        "trainer": {
            "is_resume": True,
            "pretrained_checkpoint": None,
            "max_train_steps": 0,
            "num_warmup_steps": 0,
            "save_interval": 999999,
            "eval_interval": 999999,
            "logging_frequency": 999999,
            "save_checkpoint_as_directory": True,
            "save_with_training_state": False,
            "enable_local_checkpoint_staging": False,
            "local_checkpoint_root": None,
            "temp_checkpoint_root": None,
        },
        "datasets": {
            "vla_data": {
                "per_device_batch_size": 1,
                "num_workers": 0,
                "load_all_data_for_training": False,
            }
        },
    })
    cfg = OmegaConf.merge(cfg, overrides)

    # Re-derive output_dir to match run_root_dir/run_id.
    cfg.output_dir = str(run_dir)
    cfg.config_yaml = str(run_dir / "config.full.yaml")
    return cfg


def run_ds_to_universal(input_folder: Path, output_folder: Path, timeout: int) -> None:
    """Invoke DeepSpeed's ds_to_universal conversion in a fresh subprocess."""
    python_exe = sys.executable
    cmd = [
        python_exe,
        "-m",
        "deepspeed.checkpoint.ds_to_universal",
        "--input_folder",
        str(input_folder),
        "--output_folder",
        str(output_folder),
    ]
    logger.info(f"Running ds_to_universal: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False, capture_output=False, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            f"ds_to_universal failed with exit code {result.returncode}. "
            f"stderr: {result.stderr[-2000:] if result.stderr else 'N/A'}"
        )


def main():
    args = parse_args()
    source_path = Path(args.source_checkpoint).resolve()
    output_universal_dir = Path(args.output_universal_dir).resolve()

    # Validate on every rank; lightweight but safe.
    trainer_state = validate_source_checkpoint(source_path)
    completed_steps = int(trainer_state.get("completed_steps", 0))
    tag = source_path.name  # e.g. steps_31500

    # Decide where to stage temporary data.
    if args.temp_zero_dir is not None:
        temp_zero_dir = Path(args.temp_zero_dir).resolve()
    else:
        temp_zero_dir = output_universal_dir.parent / f".tmp_zero_{tag}"

    # Use a temp directory under the repo or system temp for the fake run dir.
    temp_run_root = REPO_ROOT / "playground" / "Checkpoints"
    temp_run_root.mkdir(parents=True, exist_ok=True)

    # Only rank 0 should prepare / report the temp dirs, but everyone needs the same paths.
    run_dir = build_temp_run_dir(source_path, temp_run_root, accelerator)
    run_id = run_dir.name

    logger.info(f"Conversion temp run dir: {run_dir}")
    logger.info(f"Intermediate ZeRO checkpoint dir: {temp_zero_dir}")
    logger.info(f"Final Universal Checkpoint dir: {output_universal_dir}")

    # Load and patch config.
    cfg = load_source_config(source_path, args.config_yaml)
    cfg = patch_config_for_conversion(cfg, run_dir, run_id)
    cfg = wrap_config(cfg)

    # Replicate the minimal training setup up to prepare_training().
    _ = setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=Path(cfg.output_dir))
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    logger.info("Loading lightweight checkpoint into DeepSpeed engine...")
    trainer.prepare_training()
    logger.info("Lightweight checkpoint loaded into DeepSpeed engine.")

    # Save as DeepSpeed native ZeRO checkpoint.
    trainer.save_deepspeed_zero_checkpoint(temp_zero_dir, tag=tag)

    # Rank 0 converts the ZeRO checkpoint to Universal Checkpoint layout.
    if accelerator.is_main_process:
        if output_universal_dir.exists():
            shutil.rmtree(output_universal_dir)

        tmp_universal_dir = output_universal_dir.with_suffix(output_universal_dir.suffix + ".tmp")
        if tmp_universal_dir.exists():
            shutil.rmtree(tmp_universal_dir)

        # ds_to_universal expects *_optim_states.pt and *_model_states.pt directly
        # in the input folder, not under the DeepSpeed tag subfolder.
        zero_tag_dir = temp_zero_dir / tag
        run_ds_to_universal(
            input_folder=zero_tag_dir,
            output_folder=tmp_universal_dir,
            timeout=args.ds_to_universal_timeout,
        )

        # Atomic rename to final output.
        tmp_universal_dir.rename(output_universal_dir)
        logger.info(f"Universal Checkpoint written to {output_universal_dir}")

    accelerator.wait_for_everyone()

    # Cleanup.
    if not args.keep_temp_zero and accelerator.is_main_process:
        if temp_zero_dir.exists():
            shutil.rmtree(temp_zero_dir)
            logger.info(f"Cleaned up intermediate ZeRO checkpoint: {temp_zero_dir}")

    if accelerator.is_main_process:
        # Remove the fake run dir only; the symlink target (source checkpoint) is preserved.
        shutil.rmtree(run_dir, ignore_errors=True)
        clear_temp_run_marker(temp_run_root, accelerator)

    accelerator.wait_for_everyone()
    logger.info("Lightweight -> Universal conversion complete.")


if __name__ == "__main__":
    main()
