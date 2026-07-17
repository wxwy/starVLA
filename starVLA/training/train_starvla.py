# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).
"""

# Standard Library
import argparse
import gc
import json
import math
import os
import pickle
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Tuple

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist

# NPU support: import torch_npu and enable automatic CUDA→NPU mapping.
# On GPU-only environments this is a no-op (ImportError is silently ignored).
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    pass

import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.checkpointing import load_accelerator_state, load_custom_state
from accelerate.logging import get_logger
from accelerate.utils import (
    DeepSpeedSchedulerWrapper,
    DistributedType,
    MODEL_NAME,
    RNG_STATE_NAME,
    SAMPLER_NAME,
    SCALER_NAME,
    SCHEDULER_NAME,
)
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.dataloader import build_dataloader
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.checkpoints import (
    load_deepspeed_universal_checkpoint,
    save_deepspeed_universal_checkpoint,
)
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import (
    TrainerUtils,
    build_param_lr_groups,
    setup_optimizer_and_scheduler,
    normalize_dotlist_args,
    _is_complete_deepspeed_checkpoint_dir,
    _is_complete_deepspeed_universal_checkpoint_dir,
    _is_complete_lightweight_training_checkpoint_dir,
    save_lightweight_checkpoint_metadata,
    save_lightweight_scaler_state,
    load_lightweight_scaler_state,
)

_num_processes = int(os.environ.get("NUM_PROCESSES", "1"))
# When launched via `accelerate launch --config_file deepspeed_*.yaml`, Accelerate
# sets ACCELERATE_USE_DEEPSPEED / ACCELERATE_DEEPSPEED_CONFIG_FILE and will create
# the DeepSpeed plugin itself. In that case we must not override it with None.
_use_accelerate_deepspeed = (
    os.environ.get("ACCELERATE_USE_DEEPSPEED", "").lower() in ("true", "1")
    or os.environ.get("ACCELERATE_DEEPSPEED_CONFIG_FILE", "") != ""
)
accelerator = None

STARTUP_CHECKPOINT_STAGE_THREAD = None
STARTUP_CHECKPOINT_STAGE_ERROR = None
STARTUP_LOCAL_AHEAD_OF_NETWORK = None
STARTUP_SYNC_INFLIGHT_MARKER = None

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _get_config_path(cfg, path, default=None):
    current = cfg
    for key in path.split("."):
        try:
            current = getattr(current, key)
        except AttributeError:
            return default
    return current


def _resolve_wandb_mode(cfg) -> str:
    wandb_mode = getattr(cfg, "wandb_mode", None)
    if wandb_mode is not None:
        return str(wandb_mode).strip().lower()
    return "online"


def _wandb_mode_disables_logging(cfg) -> bool:
    mode = _resolve_wandb_mode(cfg)
    return mode == "offline" or mode.startswith("disabled")


def _touch_training_audit_config(cfg) -> None:
    """Ensure key audit fields are present in AccessTrackedConfig snapshots."""
    audit_paths = (
        "trainer.is_resume",
        "trainer.pretrained_checkpoint",
        "trainer.gradient_accumulation_steps",
        "trainer.enable_mowa_future_supervision_loss",
        "trainer.loss_scale.mowa_future_supervision",
        "trainer.enable_mowa_future_latent_prior_loss",
        "trainer.loss_scale.mowa_future_latent_prior",
        "datasets.vla_data.data_mix",
        "framework.name",
        "framework.action_model.action_model_type",
        "framework.action_model.num_target_vision_tokens",
        "framework.mowa.enable_future_supervision_loss",
        "framework.mowa.enable_layerwise_bridge_token_coupling",
        "framework.mowa.layerwise_bridge_feature_source",
        "framework.mowa.layerwise_bridge_token_intervention",
        "framework.mowa.num_bridge_tokens",
        "framework.mowa.layerwise_bridge_active_heads",
        "framework.mowa.gated_heads.enabled",
        "framework.mowa.gated_heads.comparison_scope",
        "framework.mowa.gated_heads.init_gate_value",
    )
    for path in audit_paths:
        _get_config_path(cfg, path)


def _enforce_launch_guard(cfg, *, full_path_dry_run_only: bool) -> None:
    if full_path_dry_run_only or not hasattr(cfg, "launch_guard"):
        return

    launch_guard = cfg.launch_guard
    launch_ready = bool(getattr(launch_guard, "launch_ready", False))
    requires_human_confirmation = bool(getattr(launch_guard, "requires_human_confirmation", False))
    human_confirmed = bool(getattr(launch_guard, "human_confirmed", False))
    policy_confirmed = bool(getattr(launch_guard, "policy_confirmed", False))
    if launch_ready and policy_confirmed and (not requires_human_confirmation or human_confirmed):
        return

    raise RuntimeError(
        "Training launch blocked by launch_guard: "
        f"launch_ready={launch_ready}, "
        f"policy_confirmed={policy_confirmed}, "
        f"requires_human_confirmation={requires_human_confirmation}, "
        f"human_confirmed={human_confirmed}. "
        "Set launch_ready=true, policy_confirmed=true, and, when "
        "requires_human_confirmation=true, human_confirmed=true only after explicit approval."
    )


def _build_accelerator(cfg):
    deepspeed_plugin = DeepSpeedPlugin() if (_num_processes > 1 and not _use_accelerate_deepspeed) else None
    gradient_accumulation_steps = int(getattr(cfg.trainer, "gradient_accumulation_steps", 1))
    mixed_precision = str(getattr(cfg.trainer, "mixed_precision", "no")).lower()
    if mixed_precision not in {"no", "fp16", "bf16", "fp8"}:
        mixed_precision = "no"
    return Accelerator(
        deepspeed_plugin=deepspeed_plugin,
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )

# Initialize logger
logger = get_logger(__name__)


def _parse_shard_size_to_bytes(raw_size) -> int:
    if isinstance(raw_size, int):
        return raw_size
    if isinstance(raw_size, str):
        size = raw_size.strip().upper()
        units = {
            "KB": 1024,
            "MB": 1024**2,
            "GB": 1024**3,
            "TB": 1024**4,
        }
        for unit, multiplier in units.items():
            if size.endswith(unit):
                return int(float(size[: -len(unit)].strip()) * multiplier)
        if size.isdigit():
            return int(size)
    raise ValueError(f"Unsupported checkpoint shard size: {raw_size}")


def _iter_model_state_tensors(model, *, save_frozen_backbone: bool):
    for name, param in model.named_parameters():
        if (
            not save_frozen_backbone
            and name.startswith("backbone.")
            and "lora_" not in name
        ):
            continue
        yield name, param
    for name, buffer in model.named_buffers():
        if (
            not save_frozen_backbone
            and name.startswith("backbone.")
            and "lora_" not in name
        ):
            continue
        yield name, buffer


def _tensor_num_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _streaming_save_model_shards(
    model,
    checkpoint_path: Path,
    save_format: str,
    max_shard_size,
    *,
    save_frozen_backbone: bool,
) -> None:
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    max_shard_bytes = _parse_shard_size_to_bytes(max_shard_size)
    shard_entries = []
    shard_state = {}
    shard_weight_names = []
    shard_bytes = 0
    total_size = 0

    def flush_shard():
        nonlocal shard_state, shard_weight_names, shard_bytes
        if not shard_state:
            return

        shard_idx = len(shard_entries) + 1
        if save_format == "safetensors":
            from safetensors.torch import save_file

            shard_name = f"model-{shard_idx:05d}.safetensors"
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            save_file(shard_state, str(checkpoint_path / shard_name))
            index_name = "model.safetensors.index.json"
        else:
            shard_name = f"pytorch_model-{shard_idx:05d}.bin"
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            torch.save(shard_state, checkpoint_path / shard_name)
            index_name = "pytorch_model.bin.index.json"

        shard_entries.append(
            {
                "filename": shard_name,
                "weight_names": list(shard_weight_names),
            }
        )
        shard_state.clear()
        shard_weight_names.clear()
        shard_bytes = 0
        gc.collect()
        return index_name

    index_name = None
    bare_model = model
    for name, tensor in _iter_model_state_tensors(
        bare_model,
        save_frozen_backbone=save_frozen_backbone,
    ):
        cpu_tensor = tensor.detach().to("cpu", copy=True).contiguous()
        tensor_bytes = _tensor_num_bytes(cpu_tensor)

        if shard_state and shard_bytes + tensor_bytes > max_shard_bytes:
            index_name = flush_shard()

        shard_state[name] = cpu_tensor
        shard_weight_names.append(name)
        shard_bytes += tensor_bytes
        total_size += tensor_bytes

    index_name = flush_shard() or index_name

    if index_name is None:
        raise RuntimeError(f"No tensors were saved for checkpoint: {checkpoint_path}")

    weight_map = {}
    for entry in shard_entries:
        for weight_name in entry["weight_names"]:
            weight_map[weight_name] = entry["filename"]

    index_payload = {
        "metadata": {
            "total_size": total_size,
        },
        "weight_map": weight_map,
    }
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path / index_name, "w", encoding="utf-8") as f:
        json.dump(index_payload, f, ensure_ascii=False, indent=2)
    gc.collect()


def _get_latest_checkpoint_entry(checkpoint_dir: Path):
    if not checkpoint_dir.exists():
        return None, 0

    checkpoint_entries = []
    for entry in checkpoint_dir.iterdir():
        file_match = re.match(r"steps_(\d+)_(?:pytorch_model\.pt|model\.safetensors)$", entry.name)
        dir_match = re.match(r"steps_(\d+)$", entry.name)

        if file_match and entry.is_file():
            checkpoint_entries.append((entry, int(file_match.group(1))))
        elif dir_match and entry.is_dir():
            if (
                _is_complete_deepspeed_checkpoint_dir(entry)
                or _is_complete_deepspeed_universal_checkpoint_dir(entry)
                or _is_complete_lightweight_training_checkpoint_dir(entry)
            ):
                checkpoint_entries.append((entry, int(dir_match.group(1))))

    if not checkpoint_entries:
        return None, 0

    checkpoint_entries.sort(key=lambda x: x[1])
    return checkpoint_entries[-1]


def _get_latest_checkpoint_entry_with_status(checkpoint_dir: Path):
    if not checkpoint_dir.exists():
        return None

    checkpoint_entries = []
    for entry in checkpoint_dir.iterdir():
        file_match = re.match(r"steps_(\d+)_(?:pytorch_model\.pt|model\.safetensors)$", entry.name)
        dir_match = re.match(r"steps_(\d+)$", entry.name)

        if file_match and entry.is_file():
            checkpoint_entries.append({"path": entry, "step": int(file_match.group(1)), "complete": True})
        elif dir_match and entry.is_dir():
            is_complete = (
                _is_complete_deepspeed_checkpoint_dir(entry)
                or _is_complete_deepspeed_universal_checkpoint_dir(entry)
                or _is_complete_lightweight_training_checkpoint_dir(entry)
            )
            checkpoint_entries.append({"path": entry, "step": int(dir_match.group(1)), "complete": is_complete})

    if not checkpoint_entries:
        return None

    checkpoint_entries.sort(key=lambda x: x["step"])
    return checkpoint_entries[-1]


def _get_checkpoint_status_view(checkpoint_dir: Path):
    latest_any = _get_latest_checkpoint_entry_with_status(checkpoint_dir)
    latest_complete_tuple = _get_latest_checkpoint_entry(checkpoint_dir)

    latest_complete = None
    if latest_complete_tuple[0] is not None:
        latest_complete = {
            "path": latest_complete_tuple[0],
            "step": latest_complete_tuple[1],
            "complete": True,
        }

    return latest_any, latest_complete


def _list_complete_checkpoint_entries(checkpoint_dir: Path):
    if not checkpoint_dir.exists():
        return []

    checkpoint_entries = []
    for entry in checkpoint_dir.iterdir():
        file_match = re.match(r"steps_(\d+)_(?:pytorch_model\.pt|model\.safetensors)$", entry.name)
        dir_match = re.match(r"steps_(\d+)$", entry.name)

        if file_match and entry.is_file():
            checkpoint_entries.append({"path": entry, "step": int(file_match.group(1)), "complete": True})
        elif dir_match and entry.is_dir():
            if (
                _is_complete_deepspeed_checkpoint_dir(entry)
                or _is_complete_deepspeed_universal_checkpoint_dir(entry)
                or _is_complete_lightweight_training_checkpoint_dir(entry)
            ):
                checkpoint_entries.append({"path": entry, "step": int(dir_match.group(1)), "complete": True})

    checkpoint_entries.sort(key=lambda x: x["step"])
    return checkpoint_entries


def _should_auto_resume_latest_complete(
    resume_policy: str | None,
    is_resume: bool,
    latest_checkpoint: str | Path | None,
) -> bool:
    return (
        not is_resume
        and str(resume_policy or "").strip().lower() == "resume_latest_complete_only"
        and latest_checkpoint is not None
    )


def _copy_path_for_stage(src_path: Path, dst_path: Path):
    if src_path.is_dir():
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
    else:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)


def _clear_checkpoint_entries_for_stage(checkpoint_dir: Path):
    if not checkpoint_dir.exists():
        return
    for path in checkpoint_dir.iterdir():
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


def _remove_other_checkpoint_entries_for_stage(checkpoint_dir: Path, keep_names: set[str]):
    if not checkpoint_dir.exists():
        return
    for path in checkpoint_dir.iterdir():
        if path.name in keep_names:
            continue
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


def _prune_checkpoint_entries_for_stage(checkpoint_dir: Path, keep_count: int, extra_keep_names: set[str] | None = None):
    extra_keep_names = extra_keep_names or set()
    keep_count = max(int(keep_count), 0)
    complete_entries = _list_complete_checkpoint_entries(checkpoint_dir)
    keep_names = {entry["path"].name for entry in complete_entries[-keep_count:]} if keep_count > 0 else set()
    keep_names.update(extra_keep_names)
    _remove_other_checkpoint_entries_for_stage(checkpoint_dir, keep_names)


def _apply_checkpoint_retention_policy(
    checkpoint_dir: Path,
    *,
    permanent_steps: set[int],
    keep_latest_count: int,
    strip_optimizer_from_non_latest: bool,
):
    """保留永久里程碑和最近 checkpoint，并可精简旧里程碑的 optimizer 状态。"""
    complete_entries = _list_complete_checkpoint_entries(checkpoint_dir)
    keep_latest_count = max(int(keep_latest_count), 0)
    latest_names = {
        entry["path"].name for entry in complete_entries[-keep_latest_count:]
    } if keep_latest_count else set()
    permanent_names = {
        entry["path"].name for entry in complete_entries if entry["step"] in permanent_steps
    }
    keep_names = latest_names | permanent_names

    for entry in complete_entries:
        checkpoint_path = entry["path"]
        if checkpoint_path.name not in keep_names:
            shutil.rmtree(checkpoint_path, ignore_errors=True)
            continue
        if (
            strip_optimizer_from_non_latest
            and checkpoint_path.name in permanent_names
            and checkpoint_path.name not in latest_names
        ):
            for optimizer_path in checkpoint_path.glob("optimizer_rank_*.pt"):
                optimizer_path.unlink(missing_ok=True)


def _copy_helper_artifacts_for_stage(src_dir: Path, dst_dir: Path, helper_artifact_names: tuple[str, ...]):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for artifact_name in helper_artifact_names:
        src_path = src_dir / artifact_name
        dst_path = dst_dir / artifact_name
        if src_path.exists():
            shutil.copy2(src_path, dst_path)


def _launch_background_checkpoint_sync(
    src: Path,
    dst: Path,
    log_path: Path,
    cleanup_src: bool,
    marker_path: Path | None = None,
    queue_dir: Path | None = None,
    keep_local_count: int = 1,
):
    dst.parent.mkdir(parents=True, exist_ok=True)
    queue_dir = queue_dir or (log_path.parent / ".checkpoint_sync_queue")
    queue_dir.mkdir(parents=True, exist_ok=True)

    task_path = queue_dir / f"{int(time.time() * 1e9)}_{src.name}.json"
    task_payload = {
        "src": str(src),
        "dst": str(dst),
        "cleanup_src": cleanup_src,
        "marker": "" if marker_path is None else str(marker_path),
        "keep_local_count": int(keep_local_count),
    }
    task_path.write_text(json.dumps(task_payload, ensure_ascii=False), encoding="utf-8")

    sync_script = r"""
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

queue_dir = Path(sys.argv[1])
lock_dir = Path(sys.argv[2])
log_path = Path(sys.argv[3])

def process_task(task):
    src = Path(task["src"])
    dst = Path(task["dst"])
    cleanup_src = bool(task["cleanup_src"])
    marker = Path(task["marker"]) if task.get("marker") else None
    keep_local_count = max(int(task.get("keep_local_count", 1)), 0)

    def file_signature(path):
        if not path.exists():
            return None
        if path.is_file():
            return {path.name: path.stat().st_size}
        signature = {}
        for file_path in path.rglob("*"):
            if file_path.is_file():
                signature[str(file_path.relative_to(path))] = file_path.stat().st_size
        return signature

    def paths_match(src_path, dst_path):
        return file_signature(src_path) == file_signature(dst_path)

    def prune_local_versions(current_src):
        retained_after_sync = max(keep_local_count - 1, 0)
        checkpoint_dir = current_src.parent
        pattern = re.compile(r"steps_(\d+)$")
        entries = []
        for entry in checkpoint_dir.iterdir():
            match = pattern.match(entry.name)
            if not match or not entry.is_dir():
                continue
            trainer_state = entry / "trainer_state.json"
            if trainer_state.is_file():
                entries.append((int(match.group(1)), entry))
        entries.sort(key=lambda x: x[0])
        keep_names = {entry.name for _, entry in entries[-retained_after_sync:]} if retained_after_sync > 0 else set()
        for _, entry in entries:
            if entry.name in keep_names:
                continue
            shutil.rmtree(entry, ignore_errors=True)

    if not src.exists():
        if marker is not None:
            marker.unlink(missing_ok=True)
        return

    def cleanup_after_sync():
        if cleanup_src and src.exists():
            if src.is_dir():
                shutil.rmtree(src, ignore_errors=True)
            else:
                src.unlink(missing_ok=True)
        elif src.exists() and src.is_dir():
            prune_local_versions(src)

        if marker is not None:
            marker.unlink(missing_ok=True)

    if dst.exists() and paths_match(src, dst):
        cleanup_after_sync()
        return

    tmp = dst.parent / f".{dst.name}.sync_tmp_{int(time.time() * 1e9)}"
    if src.is_dir():
        shutil.copytree(src, tmp, dirs_exist_ok=True)
    else:
        tmp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, tmp)

    stale = None
    if dst.exists():
        stale = dst.parent / f"{dst.name}.stale_{int(time.time())}"
        dst.rename(stale)

    tmp.rename(dst)
    if stale is not None and stale.exists():
        if stale.is_dir():
            shutil.rmtree(stale, ignore_errors=True)
        else:
            stale.unlink(missing_ok=True)

    cleanup_after_sync()

def lock_is_stale():
    pid_file = lock_dir / "pid"
    if not pid_file.exists():
        return True
    try:
        pid = int(pid_file.read_text().strip())
    except Exception:
        return True
    try:
        os.kill(pid, 0)
        return False
    except OSError:
        return True

try:
    lock_dir.mkdir()
except FileExistsError:
    if lock_is_stale():
        shutil.rmtree(lock_dir, ignore_errors=True)
        lock_dir.mkdir()
    else:
        sys.exit(0)

try:
    (lock_dir / "pid").write_text(str(os.getpid()), encoding="utf-8")
    while True:
        task_files = sorted(queue_dir.glob("*.json"))
        if not task_files:
            break
        task_file = task_files[0]
        with open(task_file, "r", encoding="utf-8") as f:
            task = json.load(f)
        process_task(task)
        task_file.unlink(missing_ok=True)
finally:
    shutil.rmtree(lock_dir, ignore_errors=True)
"""
    lock_dir = queue_dir / ".worker.lock"
    log_file = open(log_path, "a")
    subprocess.Popen(
        [sys.executable, "-c", sync_script, str(queue_dir), str(lock_dir), str(log_path)],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_file.close()


def _write_sync_inflight_marker(marker_path: Path, src: Path):
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_payload = {
        "source": str(src),
        "created_at": int(time.time()),
    }
    marker_path.write_text(json.dumps(marker_payload, ensure_ascii=False), encoding="utf-8")


def _clear_sync_inflight_marker(marker_path: Path):
    marker_path.unlink(missing_ok=True)


def _stage_local_checkpoint_storage(cfg):
    global STARTUP_CHECKPOINT_STAGE_ERROR, STARTUP_LOCAL_AHEAD_OF_NETWORK, STARTUP_SYNC_INFLIGHT_MARKER

    try:
        enable_local_checkpoint_staging = getattr(cfg.trainer, "enable_local_checkpoint_staging", True)
        local_checkpoint_root = (
            getattr(cfg.trainer, "local_checkpoint_root", None)
            or getattr(cfg.trainer, "temp_checkpoint_root", None)
        )
        if not enable_local_checkpoint_staging or not local_checkpoint_root:
            return

        network_output_dir = Path(cfg.run_root_dir) / cfg.run_id
        network_checkpoint_dir = network_output_dir / "checkpoints"
        local_output_dir = Path(local_checkpoint_root).expanduser() / cfg.run_id
        local_checkpoint_dir = local_output_dir / "checkpoints"
        inflight_marker_path = local_output_dir / ".startup_sync_inflight.json"
        helper_artifact_names = (
            "config.full.yaml",
            "config.yaml",
            "dataset_statistics.json",
            "summary.jsonl",
        )

        local_output_dir.mkdir(parents=True, exist_ok=True)
        local_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        local_latest_any, local_latest_complete = _get_checkpoint_status_view(local_checkpoint_dir)
        network_latest_any, network_latest_complete = _get_checkpoint_status_view(network_checkpoint_dir)

        if local_latest_any is not None and not local_latest_any["complete"]:
            if local_latest_complete is not None:
                logger.warning(
                    f"启动阶段后台检查发现本地最新 checkpoint 不完整，回退到本地最近完整版本: "
                    f"{local_latest_any['path']} -> {local_latest_complete['path']}"
                )
                _prune_checkpoint_entries_for_stage(local_checkpoint_dir, 1, {local_latest_complete["path"].name})
            else:
                logger.warning(f"启动阶段后台检查发现本地最新 checkpoint 不完整，且无完整本地版本可回退: {local_latest_any['path']}")
                _clear_checkpoint_entries_for_stage(local_checkpoint_dir)
            local_latest_any, local_latest_complete = _get_checkpoint_status_view(local_checkpoint_dir)

        local_latest_entry = local_latest_complete
        network_latest_entry = network_latest_complete

        if local_latest_entry is not None:
            _prune_checkpoint_entries_for_stage(local_checkpoint_dir, 1, {local_latest_entry["path"].name})

        if (
            local_latest_entry is not None
            and (
                network_latest_any is None
                or not network_latest_any["complete"]
                or network_latest_entry is None
                or local_latest_entry["step"] > network_latest_entry["step"]
            )
        ):
            STARTUP_LOCAL_AHEAD_OF_NETWORK = local_latest_entry["path"].name
            STARTUP_SYNC_INFLIGHT_MARKER = inflight_marker_path
            _write_sync_inflight_marker(inflight_marker_path, local_latest_entry["path"])
            target_path = network_checkpoint_dir / local_latest_entry["path"].name
            logger.info(
                f"启动阶段后台检查发现本地最新 checkpoint 新于网络盘，继续后台同步: {local_latest_entry['path']} -> {target_path}"
            )
            _launch_background_checkpoint_sync(
                local_latest_entry["path"],
                target_path,
                network_output_dir / "checkpoint_sync.log",
                cleanup_src=False,
                marker_path=inflight_marker_path,
                queue_dir=network_output_dir / ".checkpoint_sync_queue",
                keep_local_count=1,
            )
            return

        if network_latest_entry:
            network_latest_checkpoint = network_latest_entry["path"]
            network_latest_name = network_latest_checkpoint.name
            if local_latest_entry is None or local_latest_entry["path"].name != network_latest_name:
                logger.info(f"启动阶段后台检查发现本地缺少最新 checkpoint，开始复制: {network_latest_checkpoint}")
                _clear_checkpoint_entries_for_stage(local_checkpoint_dir)
                _copy_path_for_stage(network_latest_checkpoint, local_checkpoint_dir / network_latest_name)
                logger.info(f"启动阶段后台复制最新 checkpoint 完成: {network_latest_checkpoint}")
            else:
                logger.info(f"启动阶段后台检查确认本地已是最新 checkpoint: {local_latest_entry['path']}")
                _prune_checkpoint_entries_for_stage(local_checkpoint_dir, 1, {network_latest_name})
        else:
            _clear_checkpoint_entries_for_stage(local_checkpoint_dir)

        _copy_helper_artifacts_for_stage(network_output_dir, local_output_dir, helper_artifact_names)
    except Exception as exc:
        STARTUP_CHECKPOINT_STAGE_ERROR = exc


def _launch_startup_checkpoint_stage(cfg):
    global STARTUP_CHECKPOINT_STAGE_THREAD, STARTUP_CHECKPOINT_STAGE_ERROR, STARTUP_LOCAL_AHEAD_OF_NETWORK, STARTUP_SYNC_INFLIGHT_MARKER

    enable_local_checkpoint_staging = getattr(cfg.trainer, "enable_local_checkpoint_staging", True)
    local_checkpoint_root = (
        getattr(cfg.trainer, "local_checkpoint_root", None)
        or getattr(cfg.trainer, "temp_checkpoint_root", None)
    )
    if not enable_local_checkpoint_staging or not local_checkpoint_root:
        return
    if STARTUP_CHECKPOINT_STAGE_THREAD is not None:
        return

    STARTUP_CHECKPOINT_STAGE_ERROR = None
    STARTUP_LOCAL_AHEAD_OF_NETWORK = None
    STARTUP_SYNC_INFLIGHT_MARKER = None

    STARTUP_CHECKPOINT_STAGE_THREAD = threading.Thread(
        target=_stage_local_checkpoint_storage,
        args=(cfg,),
        name="startup-checkpoint-stage",
        daemon=True,
    )
    STARTUP_CHECKPOINT_STAGE_THREAD.start()


def _wait_for_startup_checkpoint_stage():
    global STARTUP_CHECKPOINT_STAGE_THREAD

    if STARTUP_CHECKPOINT_STAGE_THREAD is not None:
        STARTUP_CHECKPOINT_STAGE_THREAD.join()
        STARTUP_CHECKPOINT_STAGE_THREAD = None
    if STARTUP_CHECKPOINT_STAGE_ERROR is not None:
        raise RuntimeError(f"启动阶段后台检查本地 checkpoint 失败: {STARTUP_CHECKPOINT_STAGE_ERROR}")


def load_fast_tokenizer():
    return AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def prepare_data(cfg, accelerator, output_dir) -> DataLoader:
    """Prepare VLA training data."""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    if dist.is_initialized():
        dist.barrier()
    return vla_train_dataloader


def _is_full_path_dry_run(cfg) -> bool:
    return bool(getattr(cfg.trainer, "full_path_dry_run_only", False))


def _summarize_batch(batch) -> dict:
    if batch is None:
        return {"fetched": False}
    summary = {
        "fetched": True,
        "type": type(batch).__name__,
    }
    if isinstance(batch, (list, tuple)):
        summary["length"] = len(batch)
        if batch:
            first = batch[0]
            summary["first_item_type"] = type(first).__name__
            if isinstance(first, dict):
                summary["first_item_keys"] = sorted(str(key) for key in first.keys())
                for key in ("image", "lang", "action", "state", "mowa_future_targets", "mowa_future_masks", "mowa_future_metadata"):
                    if key in first:
                        value = first[key]
                        value_summary = {"type": type(value).__name__}
                        if hasattr(value, "shape"):
                            value_summary["shape"] = list(value.shape)
                        elif isinstance(value, (list, tuple)):
                            value_summary["length"] = len(value)
                        elif isinstance(value, dict):
                            value_summary["keys"] = sorted(str(item_key) for item_key in value.keys())
                        summary[f"first_item_{key}"] = value_summary
    elif isinstance(batch, dict):
        summary["keys"] = sorted(str(key) for key in batch.keys())
    return summary


def _summarize_forward_output(output_dict: dict | None) -> dict:
    if output_dict is None:
        return {"evaluated": False}

    summary = {"evaluated": True, "keys": sorted(str(key) for key in output_dict.keys())}
    for key, value in output_dict.items():
        if hasattr(value, "detach"):
            detached = value.detach()
            if detached.numel() == 1:
                summary[key] = float(detached.float().cpu().item())
            else:
                summary[key] = {"shape": list(detached.shape)}
        elif isinstance(value, (list, tuple)):
            summary[key] = list(value)
        elif isinstance(value, dict):
            summary[key] = sorted(str(item_key) for item_key in value.keys())
        else:
            summary[key] = value
    return summary


def _run_full_path_dry_run_forward(model, batch) -> dict:
    cuda_device = None
    if torch.cuda.is_available():
        try:
            cuda_device = next(model.parameters()).device
        except StopIteration:
            cuda_device = torch.device("cuda")
        if cuda_device.type != "cuda":
            cuda_device = torch.device("cuda")
        cuda_allocated_before = torch.cuda.memory_allocated(cuda_device)
        cuda_reserved_before = torch.cuda.memory_reserved(cuda_device)
        torch.cuda.reset_peak_memory_stats(cuda_device)
        torch.cuda.synchronize(cuda_device)
    else:
        cuda_allocated_before = None
        cuda_reserved_before = None
    start_time = time.perf_counter()
    output_dict = model(batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize(cuda_device)
    elapsed_sec = time.perf_counter() - start_time

    summary = _summarize_forward_output(output_dict)
    batch_size = len(batch) if hasattr(batch, "__len__") else None
    summary["metric_scope"] = "one_batch_no_backward_forward_dry_run"
    summary["elapsed_sec"] = float(elapsed_sec)
    summary["batch_size"] = int(batch_size) if batch_size is not None else None
    summary["samples_per_sec"] = (
        float(batch_size / elapsed_sec)
        if batch_size is not None and elapsed_sec > 0
        else None
    )
    summary["cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        cuda_allocated_after = torch.cuda.memory_allocated(cuda_device)
        cuda_reserved_after = torch.cuda.memory_reserved(cuda_device)
        cuda_peak_allocated = max(
            torch.cuda.max_memory_allocated(cuda_device),
            cuda_allocated_before,
            cuda_allocated_after,
        )
        cuda_peak_reserved = max(
            torch.cuda.max_memory_reserved(cuda_device),
            cuda_reserved_before,
            cuda_reserved_after,
        )
        summary["cuda_device"] = str(cuda_device)
        summary["cuda_device_name"] = torch.cuda.get_device_name(cuda_device)
        summary["allocated_vram_gb"] = float(cuda_allocated_after / (1024**3))
        summary["reserved_vram_gb"] = float(cuda_reserved_after / (1024**3))
        summary["peak_vram_gb"] = float(cuda_peak_allocated / (1024**3))
        summary["peak_reserved_vram_gb"] = float(cuda_peak_reserved / (1024**3))
    else:
        summary["cuda_device"] = None
        summary["cuda_device_name"] = None
        summary["allocated_vram_gb"] = None
        summary["reserved_vram_gb"] = None
        summary["peak_vram_gb"] = None
        summary["peak_reserved_vram_gb"] = None
    summary["vram_metric_scope"] = "torch_cuda_allocator_in_full_path_dry_run"
    return summary


def _load_full_path_dry_run_checkpoint(cfg, model) -> dict:
    checkpoint = getattr(cfg.trainer, "full_path_dry_run_checkpoint", None)
    requested = bool(getattr(cfg.trainer, "full_path_dry_run_load_checkpoint", False))
    if not requested:
        return {"requested": False, "loaded": False, "path": None}
    if not checkpoint:
        raise ValueError("trainer.full_path_dry_run_load_checkpoint=true requires trainer.full_path_dry_run_checkpoint")

    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path.cwd() / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"full_path_dry_run_checkpoint does not exist: {checkpoint_path}")

    start_time = time.perf_counter()
    TrainerUtils.load_pretrained_backbones(
        model,
        str(checkpoint_path),
        preferred_format=getattr(cfg.trainer, "save_format", "safetensors"),
    )
    return {
        "requested": True,
        "loaded": True,
        "path": str(checkpoint),
        "resolved_path": str(checkpoint_path),
        "elapsed_sec": float(time.perf_counter() - start_time),
    }


def _write_full_path_dry_run_report(
    cfg,
    *,
    output_dir: Path,
    model,
    dataloader,
    optimizer,
    trainer,
    batch_summary: dict | None,
    forward_summary: dict | None,
    checkpoint_load_summary: dict | None,
) -> None:
    report_path = Path(
        getattr(
            cfg.trainer,
            "full_path_dry_run_report",
            "docs_zh/mowa/mowa_e001_train_starvla_full_path_dry_run.json",
        )
    )
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total_params = sum(param.numel() for param in model.parameters())
    mowa_labels_enabled = bool(getattr(cfg.datasets.vla_data, "enable_mowa_future_labels", False))
    mowa_supervision_enabled = bool(
        getattr(
            model,
            "mowa_future_supervision_loss_enabled",
            getattr(model, "mowa_future_supervision_probe_enabled", False),
        )
    )
    mowa_supervision_active_heads = list(
        getattr(
            model,
            "mowa_future_supervision_active_heads",
            getattr(model, "mowa_future_supervision_active_heads", ()),
        )
    )
    mowa_supervision_label_status = (
        "forward_evaluated_in_full_path_dry_run"
        if (
            "mowa_future_supervision_loss" in ((forward_summary or {}).get("keys") or [])
            or "mowa_p0_supervision_loss" in ((forward_summary or {}).get("keys") or [])
        )
        else "not_evaluated_in_full_path_dry_run"
    )
    payload = {
        "stage": "full_heads",
        "experiment_id": getattr(cfg, "experiment_id", "E-001"),
        "entrypoint": "starVLA/training/train_starvla.py",
        "full_path_dry_run_only": True,
        "training_started": False,
        "checkpoint_saved": False,
        "checkpoint_loaded": bool((checkpoint_load_summary or {}).get("loaded", False)),
        "checkpoint_load": checkpoint_load_summary or {"requested": False, "loaded": False, "path": None},
        "wandb_started": False,
        "config_yaml": getattr(cfg, "config_yaml", None),
        "run_id": cfg.run_id,
        "output_dir": str(output_dir),
        "framework": {
            "name": cfg.framework.name,
            "starflow_ft_variant": getattr(cfg.framework, "starflow_ft_variant", "config_defined"),
            "action_model_type": getattr(cfg.framework.action_model, "action_model_type", None),
            "num_target_vision_tokens": getattr(cfg.framework.action_model, "num_target_vision_tokens", None),
            "state_mode": getattr(cfg.framework, "state_mode", None),
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
            "mowa_action_bridge_probe_enabled": bool(
                getattr(model, "mowa_action_bridge_probe_enabled", False)
            ),
            "mowa_layerwise_bridge_coupling_enabled": bool(
                getattr(model, "mowa_layerwise_bridge_coupling_enabled", False)
            ),
            "mowa_layerwise_bridge_coupling_status": (
                "forward_coupled_in_full_path_dry_run"
                if "mowa_layerwise_bridge_coupled" in ((forward_summary or {}).get("keys") or [])
                else "not_coupled_in_full_path_dry_run"
            ),
            "mowa_layerwise_bridge_feature_source": getattr(
                model,
                "mowa_layerwise_bridge_feature_source",
                None,
            ),
            "mowa_future_gated_heads_enabled": bool(
                getattr(model, "mowa_future_gated_heads_enabled", False)
            ),
            "mowa_layerwise_bridge_gated_heads_summary": getattr(
                model,
                "mowa_last_layerwise_bridge_gated_heads_summary",
                None,
            ),
            "mowa_future_supervision_gated_heads_summary": getattr(
                model,
                "mowa_last_future_supervision_gated_heads_summary",
                None,
            ),
            "mowa_future_supervision_probe_enabled": mowa_supervision_enabled,
            "mowa_future_supervision_active_heads": mowa_supervision_active_heads,
            "mowa_future_supervision_label_status": mowa_supervision_label_status,
        },
        "data": {
            "dataset_py": cfg.datasets.vla_data.dataset_py,
            "data_mix": cfg.datasets.vla_data.data_mix,
            "data_root_dir": str(cfg.datasets.vla_data.data_root_dir),
            "per_device_batch_size": int(cfg.datasets.vla_data.per_device_batch_size),
            "dataloader_type": type(dataloader).__name__,
            "dataloader_length": len(dataloader) if hasattr(dataloader, "__len__") else None,
            "mowa_future_labels_enabled": mowa_labels_enabled,
            "batch_summary": batch_summary or {"fetched": False},
        },
        "optimizer": {
            "type": type(optimizer).__name__,
            "param_group_count": len(optimizer.param_groups),
            "param_group_names": [group.get("name", str(index)) for index, group in enumerate(optimizer.param_groups)],
        },
        "trainer": {
            "max_train_steps": int(cfg.trainer.max_train_steps),
            "gradient_accumulation_steps": int(getattr(cfg.trainer, "gradient_accumulation_steps", 1)),
            "total_batch_size": int(trainer.total_batch_size),
            "checkpoint_format": getattr(cfg.trainer, "checkpoint_format", None),
            "save_interval": int(cfg.trainer.save_interval),
            "eval_interval": int(cfg.trainer.eval_interval),
        },
        "forward": forward_summary or {"evaluated": False},
        "go_no_go": "TBD: train_starvla full-path dry-run passed; training remains gated",
        "notes": [
            "This dry-run stops before prepare_training(), wandb, checkpoint saving, and train().",
            "Checkpoint loading is optional and only runs when trainer.full_path_dry_run_load_checkpoint is true.",
            "It validates StarVLA build/data/optimizer/trainer wiring only.",
            "MoWA future supervision probe is reported as configuration wiring; forward loss is covered by unit tests.",
            "MoWA bridge coupling into LayerwiseFM action generation is only active when the MoWA gated config enables it.",
        ],
    }
    if accelerator.is_main_process:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logger.info(f"MoWA E-001 train_starvla full-path dry-run report saved at {report_path}")


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Set optimizer and scheduler."""
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
        fused=True,
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # Strip keys unknown to transformers' get_scheduler before passing kwargs.
    sched_kwargs = {k: v for k, v in cfg.trainer.scheduler_specific_kwargs.items()}
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=sched_kwargs,
    )

    return optimizer, lr_scheduler


def _convert_deepspeed_optimizer_state_to_plain(optimizer_state, target_optimizer, model):
    """
    Convert a DeepSpeed ZeRO optimizer state dict (with flattened per-group params)
    into a standard PyTorch optimizer state dict matching ``target_optimizer``.

    DeepSpeed stores one flat ``exp_avg`` / ``exp_avg_sq`` tensor per parameter
    group and records the slice position of every original parameter in
    ``param_slice_mappings``. This function slices those flat tensors back into
    per-parameter states so the checkpoint can be resumed without DeepSpeed.
    """
    base_state = optimizer_state.get("base_optimizer_state")
    param_slice_mappings = optimizer_state.get("param_slice_mappings", [])
    if not isinstance(base_state, dict) or "param_groups" not in base_state or "state" not in base_state:
        raise RuntimeError("Invalid DeepSpeed optimizer state: missing base_optimizer_state.")

    # Build a name -> param map for all trainable parameters in the model.
    name_to_param = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            name_to_param[name] = param

    # Build a lookup from group name to target group.
    target_groups_by_name = {}
    for group in target_optimizer.param_groups:
        group_name = group.get("name")
        if group_name is None:
            raise RuntimeError("Target optimizer param group is missing the 'name' key required for DeepSpeed conversion.")
        if group_name in target_groups_by_name:
            raise RuntimeError(f"Duplicate target optimizer group name: {group_name}")
        target_groups_by_name[group_name] = group

    new_state = {}
    new_param_groups = []

    for ds_group_idx, ds_group in enumerate(base_state["param_groups"]):
        ds_group_name = ds_group.get("name", f"group_{ds_group_idx}")
        if ds_group_name not in target_groups_by_name:
            raise RuntimeError(
                f"DeepSpeed optimizer group '{ds_group_name}' not found in target optimizer. "
                f"Available groups: {list(target_groups_by_name.keys())}"
            )
        target_group = target_groups_by_name[ds_group_name]

        ds_group_state = base_state["state"][ds_group_idx]
        ds_exp_avg = ds_group_state["exp_avg"]
        ds_exp_avg_sq = ds_group_state["exp_avg_sq"]
        ds_step = ds_group_state["step"]

        slice_mapping = param_slice_mappings[ds_group_idx] if ds_group_idx < len(param_slice_mappings) else {}

        # Reconstruct per-parameter states for every parameter in the target group.
        for param in target_group["params"]:
            param_name = None
            for name, p in name_to_param.items():
                if p is param:
                    param_name = name
                    break
            if param_name is None:
                raise RuntimeError(
                    "A parameter in the target optimizer could not be matched to any named trainable model parameter."
                )
            if param_name not in slice_mapping:
                raise RuntimeError(
                    f"Parameter '{param_name}' not found in DeepSpeed param_slice_mappings for group '{ds_group_name}'."
                )
            fragment = slice_mapping[param_name]
            start = int(fragment.start)
            numel = int(fragment.numel)
            if numel != param.numel():
                raise RuntimeError(
                    f"Parameter '{param_name}' numel mismatch: checkpoint slice has {numel}, "
                    f"but model parameter has {param.numel()}."
                )
            new_state[id(param)] = {
                "step": ds_step,
                "exp_avg": ds_exp_avg[start : start + numel].view_as(param).clone(),
                "exp_avg_sq": ds_exp_avg_sq[start : start + numel].view_as(param).clone(),
            }
            # Keep Adam moments in their checkpoint dtype (normally fp32). Fused
            # AdamW requires matching devices, but downcasting exp_avg/exp_avg_sq
            # to bf16 loses optimizer precision after resume.
            for key in ("exp_avg", "exp_avg_sq"):
                new_state[id(param)][key] = new_state[id(param)][key].to(param.device)

        # Reconstruct the param group with current parameter ids but preserved hyperparameters.
        # Force fused=False because the converted state does not satisfy the strict requirements
        # of the fused AdamW kernel (contiguity / dtype / device checks can fail across params).
        new_group = {k: v for k, v in ds_group.items() if k != "params"}
        new_group["params"] = [id(p) for p in target_group["params"]]
        new_group["fused"] = False
        new_param_groups.append(new_group)

    return {"state": new_state, "param_groups": new_param_groups}


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()
        self.resume_requires_training_state = False
        self.resume_requires_lightweight_state = False
        self.runtime_metrics_report = getattr(self.config.trainer, "runtime_metrics_report", None)
        self.network_output_dir = Path(self.config.output_dir)
        self.network_checkpoint_dir = self.network_output_dir / "checkpoints"
        self.enable_local_checkpoint_staging = getattr(self.config.trainer, "enable_local_checkpoint_staging", True)
        local_checkpoint_root = (
            getattr(self.config.trainer, "local_checkpoint_root", None)
            or getattr(self.config.trainer, "temp_checkpoint_root", None)
        )
        self.local_checkpoint_root = (
            Path(local_checkpoint_root).expanduser()
            if self.enable_local_checkpoint_staging and local_checkpoint_root
            else None
        )
        self.local_output_dir = (
            self.network_output_dir if self.local_checkpoint_root is None else self.local_checkpoint_root / self.config.run_id
        )
        self.local_checkpoint_dir = self.local_output_dir / "checkpoints"
        self.helper_artifact_names = (
            "config.full.yaml",
            "config.yaml",
            "dataset_statistics.json",
            "summary.jsonl",
        )
        self.sync_log_path = self.network_output_dir / "checkpoint_sync.log"
        self.local_resume_checkpoint_paths_for_cleanup = []
        self.local_resume_cleanup_started = False
        self.save_with_training_state = getattr(self.config.trainer, "save_with_training_state", False)
        self.save_checkpoint_as_directory = getattr(self.config.trainer, "save_checkpoint_as_directory", True)
        self.checkpoint_max_shard_size = getattr(self.config.trainer, "checkpoint_max_shard_size", "5GB")
        self.save_frozen_backbone = bool(getattr(self.config.trainer, "save_frozen_backbone", False))
        self.local_checkpoint_keep_count = max(int(getattr(self.config.trainer, "local_checkpoint_keep_count", 1)), 1)
        self.checkpoint_permanent_steps = {
            int(step) for step in getattr(self.config.trainer, "checkpoint_permanent_steps", [])
        }
        self.checkpoint_keep_latest_count = max(
            int(getattr(self.config.trainer, "checkpoint_keep_latest_count", 0)), 0
        )
        self.strip_optimizer_from_non_latest_checkpoints = bool(
            getattr(self.config.trainer, "strip_optimizer_from_non_latest_checkpoints", False)
        )
        self.save_universal_checkpoint = getattr(self.config.trainer, "save_universal_checkpoint", False)
        self.checkpoint_format = self._resolve_checkpoint_format()

    def _resolve_checkpoint_format(self) -> str:
        raw_format = getattr(self.config.trainer, "checkpoint_format", None)
        if raw_format is None:
            if self.save_with_training_state:
                return "deepspeed_state"
            return "universal"

        checkpoint_format = str(raw_format).strip().lower()
        aliases = {
            "deepspeed_universal": "universal",
            "universal_full_adam": "universal",
            "deepspeed": "deepspeed_state",
            "training_state": "deepspeed_state",
            "directory": "lightweight",
        }
        checkpoint_format = aliases.get(checkpoint_format, checkpoint_format)
        valid_formats = {"lightweight", "universal", "deepspeed_state", "model_only"}
        if checkpoint_format not in valid_formats:
            raise ValueError(f"Unsupported trainer.checkpoint_format `{raw_format}`. Expected one of {sorted(valid_formats)}.")
        return checkpoint_format

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        _wait_for_startup_checkpoint_stage()
        self._setup_checkpoint_storage()
        _touch_training_audit_config(self.config)

        # Save config snapshots upfront so that even if a later setup step
        # (ckpt load / DeepSpeed init / dataloader build) crashes, the
        # produced run dir is still introspectable / from_pretrained-able.
        self._save_initial_configs()
        self._sync_helper_artifacts_to_local()

        self._init_checkpointing()
        self._adjust_lr_scheduler_for_resume()

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)

        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator,
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
        )

        if self.resume_requires_training_state and self.resume_from_checkpoint:
            self._load_checkpoint(self.resume_from_checkpoint)
        elif self.resume_requires_lightweight_state and self.resume_from_checkpoint:
            self._load_lightweight_training_state(self.resume_from_checkpoint)

        self._init_wandb()

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """Initialize Weights & Biases."""
        if _wandb_mode_disables_logging(self.config):
            logger.info("W&B disabled by wandb_mode.")
            return
        if self.accelerator.is_main_process:
            raw_wandb_run_id = getattr(self.config, "wandb_run_id", None) or self.config.run_id
            wandb_run_id = re.sub(r"[^A-Za-z0-9_.-]", "-", str(raw_wandb_run_id))[:128]
            wandb_name = getattr(self.config, "wandb_name", None) or self.config.run_id
            wandb.init(
                id=wandb_run_id,
                resume="allow",
                name=wandb_name,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
            )

    def _save_initial_configs(self):
        """Save full config and training script at the very start of training."""
        if not self.accelerator.is_main_process:
            return

        # 1. Save config.full.yaml — the complete merged config (all parameters)
        if isinstance(self.config, AccessTrackedConfig):
            full_cfg = self.config.unwrap()
        else:
            full_cfg = self.config
        for output_dir in self._all_output_dirs():
            full_yaml_path = output_dir / "config.full.yaml"
            OmegaConf.save(full_cfg, full_yaml_path, resolve=True)
            logger.info(f"📝 Full config saved at {full_yaml_path}")

            # 2. Save config.yaml — accessed-only snapshot (will be updated at checkpoints)
            if isinstance(self.config, AccessTrackedConfig):
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
                logger.info(f"📊 Accessed config snapshot saved at {output_dir / 'config.yaml'}")

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = str(self.local_checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        resume_policy = str(getattr(self.config.trainer, "resume_policy", "")).strip().lower()
        latest_checkpoint, _ = self._get_latest_checkpoint(self.checkpoint_dir)
        if _should_auto_resume_latest_complete(resume_policy, is_resume, latest_checkpoint):
            is_resume = True
            logger.info(
                "resume_policy=resume_latest_complete_only 找到完整 checkpoint，"
                f"自动恢复: {latest_checkpoint}"
            )
        self.resume_from_checkpoint = pretrained_checkpoint

        if is_resume:
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                if os.path.isdir(self.resume_from_checkpoint) and _is_complete_deepspeed_checkpoint_dir(
                    self.resume_from_checkpoint
                ):
                    self.resume_requires_training_state = True
                elif os.path.isdir(self.resume_from_checkpoint) and _is_complete_deepspeed_universal_checkpoint_dir(
                    self.resume_from_checkpoint
                ):
                    self.resume_requires_training_state = True
                elif os.path.isdir(self.resume_from_checkpoint) and _is_complete_lightweight_training_checkpoint_dir(
                    self.resume_from_checkpoint
                ):
                    self.resume_requires_lightweight_state = True
                    self.model = self.load_pretrained_backbones(
                        self.model,
                        self.resume_from_checkpoint,
                        reload_modules=None,
                        preferred_format=getattr(self.config.trainer, "save_format", "safetensors"),
                    )
                else:
                    self.model = self.load_pretrained_backbones(
                        self.model,
                        self.resume_from_checkpoint,
                        reload_modules=None,
                        preferred_format=getattr(self.config.trainer, "save_format", "safetensors"),
                    )
                logger.info(
                    f"Resuming training from checkpoint: {self.resume_from_checkpoint}, steps: {self.completed_steps}"
                )
                return

            logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
            self.completed_steps = 0

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(
                self.model,
                pretrained_checkpoint,
                reload_modules=reload_modules,
                preferred_format=getattr(self.config.trainer, "save_format", "safetensors"),
            )
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

    def _adjust_lr_scheduler_for_resume(self):
        """Adjust LR scheduler state after resuming from non-zero steps."""
        if self.resume_requires_training_state or self.resume_requires_lightweight_state:
            return
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(
                f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}"
            )

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        checkpoint_path = Path(checkpoint_path)

        if self.accelerator.distributed_type != DistributedType.DEEPSPEED or not checkpoint_path.is_dir():
            self.accelerator.load_state(str(checkpoint_path))
            self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
            return

        checkpoint_parts = self._inspect_directory_checkpoint(checkpoint_path)
        is_universal_checkpoint = _is_complete_deepspeed_universal_checkpoint_dir(checkpoint_path)
        total_stages = 2 + int(checkpoint_parts["custom_count"] > 0)

        if self.accelerator.is_main_process:
            logger.info(f"目录式 checkpoint 顶层内容: {checkpoint_parts['entries']}")
            logger.info(
                "目录式 checkpoint 检测结果: "
                f"deepspeed_tag={checkpoint_parts['deepspeed_tag']}, "
                f"is_universal={is_universal_checkpoint}, "
                f"scheduler_files={checkpoint_parts['scheduler_files']}, "
                f"sampler_files={checkpoint_parts['sampler_files']}, "
                f"rng_files={checkpoint_parts['rng_files']}, "
                f"custom_count={checkpoint_parts['custom_count']}"
            )

        for hook in self.accelerator._load_model_state_pre_hook.values():
            hook([], str(checkpoint_path))

        if is_universal_checkpoint:
            logger.info(f"[1/{total_stages}] 开始加载 DeepSpeed Universal 模型/优化器状态: {checkpoint_path}")
            load_path, _ = load_deepspeed_universal_checkpoint(
                self.model,
                checkpoint_path,
                load_optimizer_states=True,
                load_lr_scheduler_states=True,
            )
        else:
            logger.info(
                f"[1/{total_stages}] 开始加载 DeepSpeed 模型/优化器状态: "
                f"{checkpoint_path}, tag={checkpoint_parts['deepspeed_tag']}"
            )
            load_path, _ = self.model.load_checkpoint(
                str(checkpoint_path),
                checkpoint_parts["deepspeed_tag"],
                load_module_strict=True,
                load_optimizer_states=True,
                load_lr_scheduler_states=True,
            )
        if load_path is None:
            raise RuntimeError(f"DeepSpeed checkpoint load failed: {checkpoint_path}")
        logger.info(f"[1/{total_stages}] DeepSpeed 模型/优化器状态加载完成: {load_path}")

        if checkpoint_parts["scheduler_files"] or checkpoint_parts["rng_files"] or checkpoint_parts["sampler_files"]:
            logger.info(f"[2/{total_stages}] 开始加载 scheduler / dataloader / RNG 状态")
            schedulers = [scheduler for scheduler in self.accelerator._schedulers if not isinstance(scheduler, DeepSpeedSchedulerWrapper)]
            override_attributes = load_accelerator_state(
                str(checkpoint_path),
                [],
                [],
                schedulers,
                self.accelerator._dataloaders,
                self.accelerator.state.process_index,
                self.accelerator.scaler,
                "cpu",
            )
            if "step" in override_attributes:
                self.accelerator.step = override_attributes["step"]
            logger.info(
                f"[2/{total_stages}] scheduler / dataloader / RNG 状态加载完成"
            )
        else:
            logger.warning(f"[2/{total_stages}] checkpoint 不包含 scheduler / dataloader / RNG 状态，跳过该阶段")

        if checkpoint_parts["custom_count"] > 0:
            logger.info(f"[3/{total_stages}] 开始加载 {checkpoint_parts['custom_count']} 个自定义状态")
            for index, obj in enumerate(self.accelerator._custom_objects):
                load_custom_state(obj, str(checkpoint_path), index)
            logger.info(f"[3/{total_stages}] 自定义状态加载完成")

        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _load_lightweight_training_state(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        total_stages = 4
        with open(checkpoint_path / "trainer_state.json", "r", encoding="utf-8") as f:
            trainer_state = json.load(f)

        logger.info(f"[1/{total_stages}] 开始加载 lightweight optimizer 状态")
        optimizer_state_path = self._get_lightweight_optimizer_state_path(checkpoint_path, trainer_state)
        if optimizer_state_path is None:
            logger.warning(
                "跳过 lightweight optimizer 状态恢复：checkpoint 的 optimizer 分片与当前 world_size 不兼容，"
                "将使用新初始化的 optimizer 状态继续训练。"
            )
        else:
            logger.info(f"[1/{total_stages}] lightweight optimizer 状态文件: {optimizer_state_path}")
            optimizer_state = torch.load(
                optimizer_state_path,
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
            is_deepspeed_optimizer_state = (
                isinstance(optimizer_state, dict)
                and ("base_optimizer_state" in optimizer_state or "zero_stage" in optimizer_state or "ds_version" in optimizer_state)
            )
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED and hasattr(self.optimizer, "optimizer"):
                world_size = self.accelerator.num_processes
                state_dict_list = [None] * world_size
                state_dict_list[self.accelerator.process_index] = optimizer_state
                self.optimizer.optimizer.load_state_dict(
                    state_dict_list,
                    load_optimizer_states=True,
                    load_from_fp32_weights=False,
                )
            elif is_deepspeed_optimizer_state:
                # Fallback: load the underlying PyTorch optimizer state from a DeepSpeed
                # checkpoint even though the current run is not using DeepSpeed.
                # This preserves momentum/variance and param_groups so that a run originally
                # trained with DeepSpeed can be resumed on a single GPU without DeepSpeed.
                logger.warning(
                    f"Checkpoint optimizer state at {optimizer_state_path} was saved under DeepSpeed, "
                    f"but the current distributed_type is {self.accelerator.distributed_type}. "
                    f"Extracting base_optimizer_state and loading it without DeepSpeed. "
                    f"DeepSpeed-specific states (loss scaler, fp32 partitions) will be discarded."
                )
                base_optimizer_state = optimizer_state.get("base_optimizer_state")
                if not isinstance(base_optimizer_state, dict) or "param_groups" not in base_optimizer_state:
                    raise RuntimeError(
                        f"Cannot downgrade-load DeepSpeed optimizer state: missing base_optimizer_state "
                        f"or param_groups in {optimizer_state_path}."
                    )
                plain_optimizer_state = _convert_deepspeed_optimizer_state_to_plain(
                    optimizer_state, self.optimizer, self.model
                )
                self.optimizer.load_state_dict(plain_optimizer_state)
            else:
                self.optimizer.load_state_dict(optimizer_state)
                # The underlying AdamW was created with fused=True. Old checkpoints
                # saved during fused=False runs persist fused=False in param_groups.
                # For plain (non-DeepSpeed) checkpoints, move optimizer state
                # tensors to the same device as their parameters and restore
                # fused=True. Preserve Adam state dtype, which should remain fp32
                # even when model parameters are bf16.
                for group in self.optimizer.param_groups:
                    for p in group["params"]:
                        if p in self.optimizer.state:
                            state = self.optimizer.state[p]
                            for key in ("exp_avg", "exp_avg_sq", "step"):
                                if key in state and isinstance(state[key], torch.Tensor):
                                    state[key] = state[key].to(p.device)
                    group["fused"] = True
            del optimizer_state
        logger.info(f"[1/{total_stages}] lightweight optimizer 状态加载完成")

        logger.info(f"[2/{total_stages}] 开始加载 lightweight scheduler / trainer 状态")
        scheduler_state = torch.load(
            checkpoint_path / "scheduler.pt",
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        self.lr_scheduler.load_state_dict(scheduler_state)
        del scheduler_state

        self.completed_steps = int(trainer_state.get("completed_steps", self.completed_steps))
        del trainer_state
        gc.collect()
        logger.info(f"[2/{total_stages}] lightweight scheduler / trainer 状态加载完成")

        logger.info(f"[3/{total_stages}] 开始恢复 lightweight scaler 状态")
        self._load_lightweight_scaler_state(checkpoint_path)
        logger.info(f"[3/{total_stages}] lightweight scaler 状态恢复完成")

        logger.info(f"[4/{total_stages}] 开始恢复 lightweight RNG 状态")
        self._load_lightweight_rng_state(checkpoint_path)
        logger.info(f"[4/{total_stages}] lightweight RNG 状态恢复完成")

        self.accelerator.print(f"Resumed lightweight training state from checkpoint: {checkpoint_path}")

    def _get_lightweight_optimizer_state_path(self, checkpoint_path: Path, trainer_state: dict) -> Path | None:
        rank = self.accelerator.process_index
        rank_state_path = checkpoint_path / f"optimizer_rank_{rank:05d}.pt"
        optimizer_format = trainer_state.get("optimizer_format")
        saved_world_size = trainer_state.get("optimizer_world_size")
        current_world_size = self.accelerator.num_processes

        if optimizer_format == "rank_sharded":
            if int(saved_world_size) != current_world_size:
                logger.warning(
                    f"lightweight optimizer world_size 不匹配: saved={saved_world_size}, current={current_world_size}"
                )
                return None
            if not rank_state_path.exists():
                raise FileNotFoundError(
                    f"Missing lightweight optimizer state for rank {rank}: {rank_state_path}"
                )
            return rank_state_path

        legacy_state_path = checkpoint_path / "optimizer.pt"
        if legacy_state_path.exists():
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED and current_world_size != 1:
                logger.warning(
                    f"旧版单文件 optimizer.pt 只能可靠恢复到 world_size=1，当前 world_size={current_world_size}"
                )
                return None
            return legacy_state_path

        raise FileNotFoundError(
            f"Missing lightweight optimizer state for rank {rank}: {rank_state_path}"
        )

    @staticmethod
    def _inspect_directory_checkpoint(checkpoint_path: Path) -> dict:
        entries = sorted(path.name for path in checkpoint_path.iterdir())
        latest_file = checkpoint_path / "latest"
        deepspeed_tag = MODEL_NAME
        if latest_file.exists():
            deepspeed_tag = latest_file.read_text().strip() or MODEL_NAME

        scheduler_files = [name for name in entries if name.startswith(SCHEDULER_NAME)]
        scaler_files = [name for name in entries if name.startswith(SCALER_NAME)]
        sampler_files = [name for name in entries if name.startswith(SAMPLER_NAME) or name.startswith("dl_state_dict")]
        rng_files = [name for name in entries if name.startswith(RNG_STATE_NAME)]
        custom_files = [name for name in entries if name.startswith("custom_checkpoint_")]

        return {
            "entries": entries,
            "deepspeed_tag": deepspeed_tag,
            "scheduler_files": scheduler_files,
            "scaler_files": scaler_files,
            "sampler_files": sampler_files,
            "rng_files": rng_files,
            "custom_count": len(custom_files),
        }

    def _save_checkpoint(self):
        """Save current training state."""
        save_format = getattr(self.config.trainer, "save_format", "safetensors")
        checkpoint_path = self.local_checkpoint_dir / f"steps_{self.completed_steps}"
        self._ensure_local_checkpoint_capacity(checkpoint_path)

        if self.checkpoint_format == "universal":
            self._save_deepspeed_universal_checkpoint(checkpoint_path, save_format)
        elif self.checkpoint_format == "deepspeed_state":
            self.accelerator.save_state(output_dir=str(checkpoint_path), safe_serialization=(save_format == "safetensors"))
        elif self.checkpoint_format == "lightweight":
            self._save_lightweight_directory_checkpoint(checkpoint_path, save_format)
        elif self.accelerator.is_main_process and save_format == "safetensors":
            from safetensors.torch import save_file

            state_dict = self.accelerator.get_state_dict(self.model)
            save_file(state_dict, str(checkpoint_path) + "_model.safetensors")
            del state_dict
            gc.collect()
        elif self.accelerator.is_main_process and save_format == "pt":
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, str(checkpoint_path) + "_pytorch_model.pt")
            del state_dict
            gc.collect()
        elif save_format not in {"pt", "safetensors"}:
            raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            self._append_summary_entry({"steps": self.completed_steps})
            self._sync_accessed_config_snapshots()
            if self.checkpoint_permanent_steps or self.checkpoint_keep_latest_count:
                _apply_checkpoint_retention_policy(
                    self.local_checkpoint_dir,
                    permanent_steps=self.checkpoint_permanent_steps,
                    keep_latest_count=self.checkpoint_keep_latest_count,
                    strip_optimizer_from_non_latest=self.strip_optimizer_from_non_latest_checkpoints,
                )
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")
            self._enqueue_checkpoint_sync(checkpoint_path)

        self.accelerator.wait_for_everyone()

    def save_deepspeed_zero_checkpoint(self, output_dir, tag=None):
        """Save the current DeepSpeed engine state as a native ZeRO checkpoint.

        This is intended for one-shot conversion from lightweight checkpoints to
        DeepSpeed Universal Checkpoints. It should only be called when the model
        has already been prepared as a DeepSpeed engine.
        """
        if self.accelerator.distributed_type != DistributedType.DEEPSPEED:
            raise RuntimeError("save_deepspeed_zero_checkpoint requires a DeepSpeed engine")

        output_dir = Path(output_dir)
        if self.accelerator.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
        self.accelerator.wait_for_everyone()

        if tag is None:
            tag = f"steps_{self.completed_steps}"

        logger.info(f"Saving DeepSpeed ZeRO checkpoint to {output_dir} with tag {tag}")
        self.model.save_checkpoint(str(output_dir), tag=tag)
        self.accelerator.wait_for_everyone()
        logger.info(f"DeepSpeed ZeRO checkpoint saved to {output_dir / tag}")

    def _save_deepspeed_universal_checkpoint(self, checkpoint_path: Path, save_format: str):
        """Save a full Adam DeepSpeed Universal Checkpoint.

        The lightweight checkpoint is used only as a temporary source for
        rank-sharded ZeRO optimizer states. Adam exp_avg/exp_avg_sq are
        preserved from optimizer_rank_*.pt files.
        """
        if self.accelerator.distributed_type != DistributedType.DEEPSPEED:
            raise RuntimeError("save_universal_checkpoint requires DeepSpeed training")
        if save_format != "safetensors":
            raise ValueError("save_universal_checkpoint currently requires trainer.save_format=safetensors")

        save_deepspeed_universal_checkpoint(
            checkpoint_path=checkpoint_path,
            save_format=save_format,
            model=self.model,
            accelerator=self.accelerator,
            save_lightweight_checkpoint=self._save_lightweight_directory_checkpoint,
            logger=logger,
        )

    def _log_metrics(self, metrics):
        """Record training metrics."""
        if self.completed_steps % self.config.trainer.logging_frequency == 0 and self.accelerator.is_main_process:
            last_lrs = self.lr_scheduler.get_last_lr()
            for i, group in enumerate(self.optimizer.param_groups):
                group_name = group.get("name", str(i))
                metrics[f"learning_rate/{group_name}"] = last_lrs[i] if i < len(last_lrs) else last_lrs[-1]
            metrics["epoch"] = round(
                self.completed_steps * self.accelerator.gradient_accumulation_steps / len(self.vla_train_dataloader),
                2,
            )
            if not _wandb_mode_disables_logging(self.config):
                wandb.log(metrics, step=self.completed_steps)
            logger.info(f"Step {self.completed_steps}, Loss: {metrics})")
            self._write_runtime_metrics(metrics)
            self._maybe_cleanup_local_resume_checkpoint()

    def _write_runtime_metrics(self, metrics):
        if not self.runtime_metrics_report:
            return
        report_path = Path(self.runtime_metrics_report)
        if not report_path.is_absolute():
            report_path = Path.cwd() / report_path
        report_path.parent.mkdir(parents=True, exist_ok=True)
        timing_data = float(metrics.get("timing/data", 0.0))
        timing_model = float(metrics.get("timing/model", 0.0))
        step_time = timing_data + timing_model
        payload = {
            "completed_steps": int(self.completed_steps),
            "total_batch_size": int(self.total_batch_size),
            "timing_data_sec": timing_data,
            "timing_model_sec": timing_model,
            "step_time_sec": step_time,
            "samples_per_sec": float(self.total_batch_size / step_time) if step_time > 0 else None,
            "metrics": {
                key: float(value)
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            },
        }
        if torch.cuda.is_available():
            payload["cuda_device_name"] = torch.cuda.get_device_name()
            payload["peak_vram_gb"] = float(torch.cuda.max_memory_allocated() / (1024**3))
            payload["peak_reserved_gb"] = float(torch.cuda.max_memory_reserved() / (1024**3))
        with report_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _create_data_iterators(self):
        """Create data iterators."""
        self.vla_iter = iter(self.vla_train_dataloader)

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)."""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def train(self):
        """Execute training loop."""
        self._log_training_config()
        if self.runtime_metrics_report and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._create_data_iterators()
        progress_bar = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            if self.accelerator.sync_gradients and self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics)

            step_metrics["timing/data"] = t_end_data - t_start_data
            step_metrics["timing/model"] = t_end_model - t_start_model
            if self.accelerator.sync_gradients:
                self._log_metrics(step_metrics)

            if (
                self.accelerator.sync_gradients
                and self.completed_steps % self.config.trainer.save_interval == 0
                and self.completed_steps > 0
            ):
                self._save_checkpoint()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """Run simple action-eval over multiple batches and attach score to metrics.

        The number of eval batches is read from ``config.trainer.eval_num_batches``
        (default 1).  When per_device_batch_size is small (e.g. 1 on a 4090) set
        this to 8–16 to get a stable mse_score estimate.
        """
        eval_num_batches = getattr(self.config.trainer, "eval_num_batches", 1)
        total_score = 0.0
        total_pots = 0

        for _ in range(eval_num_batches):
            examples = self._get_next_batch()
            actions = [example["action"] for example in examples]
            output_dict = self.accelerator.unwrap_model(self.model).predict_action(
                examples=examples, use_ddim=True, num_ddim_steps=20
            )

            if self.accelerator.is_main_process:
                normalized_actions = output_dict["normalized_actions"]
                actions_np = np.array(actions)
                num_pots = int(np.prod(actions_np.shape))
                score = TrainerUtils.euclidean_distance(normalized_actions, actions_np)
                total_score += score
                total_pots += num_pots

            del examples, actions, output_dict

        if self.accelerator.is_main_process and total_pots > 0:
            step_metrics["mse_score"] = total_score / total_pots
            step_metrics["eval_num_samples"] = total_pots

        if dist.is_initialized():
            dist.barrier()
        return step_metrics

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  is_resume = {_get_config_path(self.config, 'trainer.is_resume', None)}")
            logger.info(
                f"  enable_mowa_future_supervision_loss = "
                f"{_get_config_path(self.config, 'trainer.enable_mowa_future_supervision_loss', False)}"
            )
            logger.info(f"  resume_from_checkpoint = {getattr(self, 'resume_from_checkpoint', None)}")
            logger.info(
                f"  trainer.pretrained_checkpoint = "
                f"{_get_config_path(self.config, 'trainer.pretrained_checkpoint', None)}"
            )
            logger.info(f"  data_mix = {_get_config_path(self.config, 'datasets.vla_data.data_mix', None)}")
            logger.info(f"  framework.name = {_get_config_path(self.config, 'framework.name', None)}")
            logger.info(
                f"  action_model_type = "
                f"{_get_config_path(self.config, 'framework.action_model.action_model_type', None)}"
            )
            logger.info(
                f"  num_target_vision_tokens = "
                f"{_get_config_path(self.config, 'framework.action_model.num_target_vision_tokens', None)}"
            )
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_vla, batch_vlm=None):
        """Execute single training step."""
        with self.accelerator.accumulate(self.model):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
                total_loss = action_loss
                mowa_future_supervision_loss = output_dict.get("mowa_future_supervision_loss")
                if mowa_future_supervision_loss is None:
                    mowa_future_supervision_loss = output_dict.get("mowa_p0_supervision_loss")
                mowa_future_latent_prior_loss = output_dict.get("mowa_future_latent_prior_loss")
                if (
                    bool(getattr(self.config.trainer, "enable_mowa_future_supervision_loss", False))
                    and mowa_future_supervision_loss is not None
                ):
                    total_loss = total_loss + (
                        mowa_future_supervision_loss
                        * float(getattr(self.config.trainer.loss_scale, "mowa_future_supervision", 1.0))
                    )

                multiview_metrics = {}
                if output_dict.get("loss_future_main") is not None:
                    mowa_cfg = getattr(self.config.framework, "mowa", None)
                    future_loss_cfg = getattr(mowa_cfg, "future_loss", None)
                    done_head_cfg = getattr(mowa_cfg, "done_head", None)
                    prior_scale = float(
                        getattr(self.config.trainer.loss_scale, "mowa_future_latent_prior", 1.0)
                    )
                    if not bool(getattr(self.config.trainer, "enable_mowa_future_latent_prior_loss", False)):
                        prior_scale = 0.0
                    main_weight = float(getattr(future_loss_cfg, "main_weight", 1.0))
                    wrist_weight = float(getattr(future_loss_cfg, "wrist_weight", 1.0))
                    done_weight = float(getattr(done_head_cfg, "loss_weight", 1.0))
                    for name in (
                        "loss_future_main",
                        "loss_future_wrist",
                        "loss_future_total",
                        "loss_done",
                        "loss_multiview_total",
                        "cross_view_gate_mean",
                        "cross_view_output_norm",
                        "main_future_pred_norm",
                        "wrist_future_pred_norm",
                    ):
                        value = output_dict.get(name)
                        if value is not None:
                            multiview_metrics[name] = float(value.detach().float().item())
                    multiview_metrics["loss_action"] = float(action_loss.detach().float().item())
                    weighted_future_main = prior_scale * main_weight * output_dict["loss_future_main"]
                    weighted_future_wrist = prior_scale * wrist_weight * output_dict["loss_future_wrist"]
                    weighted_done = prior_scale * done_weight * output_dict["loss_done"]
                    weighted_future_total = weighted_future_main + weighted_future_wrist + weighted_done
                    multiview_metrics["weighted_future_main"] = float(
                        weighted_future_main.detach().float().item()
                    )
                    multiview_metrics["weighted_future_wrist"] = float(
                        weighted_future_wrist.detach().float().item()
                    )
                    multiview_metrics["weighted_done"] = float(weighted_done.detach().float().item())
                    multiview_metrics["weighted_future_total"] = float(
                        weighted_future_total.detach().float().item()
                    )
                    multiview_metrics["aux_to_action_ratio"] = float(
                        (weighted_future_total.detach().float() / action_loss.detach().float().clamp_min(1e-8))
                        .item()
                    )
                    for layer, value in output_dict.get("cross_view_gate_by_layer", {}).items():
                        multiview_metrics[f"cross_view_gate/layer_{layer}"] = float(
                            value.detach().float().item()
                        )
                if (
                    bool(getattr(self.config.trainer, "enable_mowa_future_latent_prior_loss", False))
                    and mowa_future_latent_prior_loss is not None
                ):
                    total_loss = total_loss + (
                        mowa_future_latent_prior_loss
                        * float(getattr(self.config.trainer.loss_scale, "mowa_future_latent_prior", 1.0))
                    )
                if multiview_metrics:
                    multiview_metrics["loss_total"] = float(total_loss.detach().float().item())

            action_loss_item = action_loss.item()
            if not hasattr(self, "_loss_accum"):
                self._loss_accum = {
                    "action_dit_loss": 0.0,
                    "mowa_future_supervision_loss": 0.0,
                    "mowa_future_latent_prior_loss": 0.0,
                    "count": 0,
                }
            self._loss_accum["action_dit_loss"] += action_loss_item
            if mowa_future_supervision_loss is not None:
                self._loss_accum["mowa_future_supervision_loss"] += mowa_future_supervision_loss.item()
            if mowa_future_latent_prior_loss is not None:
                self._loss_accum["mowa_future_latent_prior_loss"] += mowa_future_latent_prior_loss.item()
            self._loss_accum["count"] += 1

            self.accelerator.backward(total_loss)
            if multiview_metrics:
                cross_view_grad_sq = sum(
                    float(parameter.grad.detach().float().pow(2).sum().item())
                    for name, parameter in self.model.named_parameters()
                    if "cross_view_adapters" in name and parameter.grad is not None
                )
                if cross_view_grad_sq > 0:
                    multiview_metrics["cross_view_grad_norm"] = cross_view_grad_sq**0.5

            if self.accelerator.sync_gradients and self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            self.optimizer.step()
            # Only step the LR scheduler when gradients are actually synced
            # (i.e., not mid-accumulation). Without this guard the scheduler
            # runs gradient_accumulation_steps times faster than intended,
            # causing warmup to end too early and cosine decay to bottom out
            # at min_lr well before max_train_steps is reached.
            if self.accelerator.sync_gradients:
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

            if self.accelerator.sync_gradients:
                metrics = {
                    "action_dit_loss": self._loss_accum["action_dit_loss"] / max(self._loss_accum["count"], 1),
                    "mowa_future_supervision_loss": self._loss_accum["mowa_future_supervision_loss"]
                    / max(self._loss_accum["count"], 1),
                    "mowa_future_latent_prior_loss": self._loss_accum["mowa_future_latent_prior_loss"]
                    / max(self._loss_accum["count"], 1),
                    "action_dit_loss_last_micro": action_loss_item,
                    "train_accumulation_micro_steps": self._loss_accum["count"],
                }
                metrics.update(multiview_metrics)
                self._loss_accum = {
                    "action_dit_loss": 0.0,
                    "mowa_future_supervision_loss": 0.0,
                    "mowa_future_latent_prior_loss": 0.0,
                    "count": 0,
                }
                return metrics

        return {
            "action_dit_loss_last_micro": action_loss_item,
        }

    def _finalize_training(self):
        """Training end processing."""
        if bool(getattr(self.config.trainer, "skip_final_checkpoint", False)):
            logger.info("Final checkpoint skipped because trainer.skip_final_checkpoint=true.")
            return
        save_format = getattr(self.config.trainer, "save_format", "safetensors")
        final_checkpoint = self.local_output_dir / "final_model"
        self._ensure_local_checkpoint_capacity(final_checkpoint)
        os.makedirs(final_checkpoint, exist_ok=True)

        if self.checkpoint_format == "universal":
            self._save_deepspeed_universal_checkpoint(final_checkpoint, save_format)
        elif self.checkpoint_format == "deepspeed_state":
            self.accelerator.save_state(output_dir=str(final_checkpoint), safe_serialization=(save_format == "safetensors"))
        elif self.checkpoint_format == "lightweight":
            self._save_lightweight_directory_checkpoint(final_checkpoint, save_format)
        elif self.accelerator.is_main_process and save_format == "safetensors":
            from safetensors.torch import save_file

            state_dict = self.accelerator.get_state_dict(self.model)
            save_file(state_dict, str(final_checkpoint / "model.safetensors"))
            del state_dict
            gc.collect()
        elif self.accelerator.is_main_process and save_format == "pt":
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, str(final_checkpoint / "pytorch_model.pt"))
            del state_dict
            gc.collect()
        elif save_format not in {"pt", "safetensors"}:
            raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")
            self._enqueue_checkpoint_sync(final_checkpoint)

        if self.accelerator.is_main_process and not _wandb_mode_disables_logging(self.config):
            wandb.finish()

        self.accelerator.wait_for_everyone()

    def _all_output_dirs(self):
        if self.local_output_dir == self.network_output_dir:
            return [self.network_output_dir]
        return [self.network_output_dir, self.local_output_dir]

    def _setup_checkpoint_storage(self):
        global STARTUP_LOCAL_AHEAD_OF_NETWORK, STARTUP_SYNC_INFLIGHT_MARKER

        if self.local_output_dir == self.network_output_dir:
            return

        self.local_output_dir.mkdir(parents=True, exist_ok=True)
        self.local_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if self.accelerator.is_main_process:
            local_latest_any, local_latest_complete = _get_checkpoint_status_view(self.local_checkpoint_dir)
            network_latest_any, network_latest_complete = _get_checkpoint_status_view(self.network_checkpoint_dir)

            if local_latest_any is not None and not local_latest_any["complete"]:
                if local_latest_complete is not None:
                    logger.warning(
                        f"本地临时路径最新 checkpoint 不完整，回退到本地最近完整版本: "
                        f"{local_latest_any['path']} -> {local_latest_complete['path']}"
                    )
                    self._prune_checkpoint_entries(self.local_checkpoint_dir, 1, {local_latest_complete["path"].name})
                else:
                    logger.warning(f"本地临时路径最新 checkpoint 不完整，且无完整本地版本可回退: {local_latest_any['path']}")
                    self._clear_checkpoint_entries(self.local_checkpoint_dir)
                local_latest_any, local_latest_complete = _get_checkpoint_status_view(self.local_checkpoint_dir)

            local_latest_entry = local_latest_complete
            network_latest_entry = network_latest_complete

            if local_latest_entry is not None:
                self._prune_checkpoint_entries(self.local_checkpoint_dir, 1, {local_latest_entry["path"].name})

            if (
                local_latest_entry is not None
                and (
                    network_latest_any is None
                    or not network_latest_any["complete"]
                    or network_latest_entry is None
                    or local_latest_entry["step"] > network_latest_entry["step"]
                )
            ):
                logger.info(f"本地临时路径最新 checkpoint 新于网络盘，直接复用: {local_latest_entry['path']}")
                self.local_resume_checkpoint_paths_for_cleanup = [local_latest_entry["path"]]
                STARTUP_LOCAL_AHEAD_OF_NETWORK = local_latest_entry["path"].name
                STARTUP_SYNC_INFLIGHT_MARKER = self.local_output_dir / ".startup_sync_inflight.json"
            elif network_latest_entry:
                network_latest_checkpoint = network_latest_entry["path"]
                network_latest_name = network_latest_checkpoint.name
                if local_latest_entry is None or local_latest_entry["path"].name != network_latest_name:
                    logger.info(f"本地临时路径缺少最新 checkpoint，开始从网络盘引导: {network_latest_checkpoint}")
                    self._clear_checkpoint_entries(self.local_checkpoint_dir)
                    self._copy_path(network_latest_checkpoint, self.local_checkpoint_dir / network_latest_name)
                    logger.info(f"已复制最新 checkpoint 到本地: {network_latest_checkpoint}")
                else:
                    logger.info(f"本地临时路径已存在最新 checkpoint，直接复用: {local_latest_entry['path']}")
                    self._prune_checkpoint_entries(self.local_checkpoint_dir, 1, {network_latest_name})

                self.local_resume_checkpoint_paths_for_cleanup = [
                    path for path in self.local_checkpoint_dir.iterdir() if path.name == network_latest_name
                ]
            else:
                self._clear_checkpoint_entries(self.local_checkpoint_dir)
                self.local_resume_checkpoint_paths_for_cleanup = []

            if STARTUP_LOCAL_AHEAD_OF_NETWORK is None:
                self._copy_helper_artifacts(self.network_output_dir, self.local_output_dir)
        self.accelerator.wait_for_everyone()

    def _sync_helper_artifacts_to_local(self):
        if self.local_output_dir == self.network_output_dir or not self.accelerator.is_main_process:
            return
        self._copy_helper_artifacts(self.network_output_dir, self.local_output_dir)

    def _copy_helper_artifacts(self, src_dir: Path, dst_dir: Path):
        dst_dir.mkdir(parents=True, exist_ok=True)
        for artifact_name in self.helper_artifact_names:
            src_path = src_dir / artifact_name
            dst_path = dst_dir / artifact_name
            if src_path.exists():
                shutil.copy2(src_path, dst_path)

    @staticmethod
    def _copy_path(src_path: Path, dst_path: Path):
        if src_path.is_dir():
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)

    @staticmethod
    def _path_size_bytes(path: Path) -> int:
        if not path.exists():
            return 0
        if path.is_file():
            return path.stat().st_size
        return sum(sub_path.stat().st_size for sub_path in path.rglob("*") if sub_path.is_file())

    @staticmethod
    def _clear_checkpoint_entries(checkpoint_dir: Path):
        for path in checkpoint_dir.iterdir():
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)

    @staticmethod
    def _remove_other_checkpoint_entries(checkpoint_dir: Path, keep_names: set[str]):
        for path in checkpoint_dir.iterdir():
            if path.name in keep_names:
                continue
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)

    @staticmethod
    def _prune_checkpoint_entries(checkpoint_dir: Path, keep_count: int, extra_keep_names: set[str] | None = None):
        extra_keep_names = extra_keep_names or set()
        keep_count = max(int(keep_count), 0)
        complete_entries = _list_complete_checkpoint_entries(checkpoint_dir)
        keep_names = {entry["path"].name for entry in complete_entries[-keep_count:]} if keep_count > 0 else set()
        keep_names.update(extra_keep_names)
        VLATrainer._remove_other_checkpoint_entries(checkpoint_dir, keep_names)

    def _append_summary_entry(self, summary_data: dict):
        for output_dir in self._all_output_dirs():
            with open(output_dir / "summary.jsonl", "a") as f:
                f.write(json.dumps(summary_data) + "\n")

    def _save_lightweight_directory_checkpoint(self, checkpoint_path: Path, save_format: str):
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        if self.accelerator.is_main_process:
            bare_model = self.accelerator.unwrap_model(self.model)
            _streaming_save_model_shards(
                bare_model,
                checkpoint_path,
                save_format,
                self.checkpoint_max_shard_size,
                save_frozen_backbone=self.save_frozen_backbone,
            )
            self._save_wan_lora_adapter(bare_model, checkpoint_path)
            gc.collect()

            scheduler_state = self.lr_scheduler.state_dict()
            torch.save(scheduler_state, checkpoint_path / "scheduler.pt")
            del scheduler_state
            gc.collect()

            self._save_lightweight_scaler_state(checkpoint_path)
            self._save_lightweight_checkpoint_metadata(checkpoint_path)

            trainer_state = {
                "completed_steps": self.completed_steps,
                "save_format": save_format,
                "checkpoint_type": "lightweight_training",
                "optimizer_format": "rank_sharded",
                "optimizer_world_size": self.accelerator.num_processes,
                "save_frozen_backbone": self.save_frozen_backbone,
                "omitted_model_state_prefixes": [] if self.save_frozen_backbone else ["backbone."],
            }
            with open(checkpoint_path / "trainer_state.json", "w", encoding="utf-8") as f:
                json.dump(trainer_state, f, ensure_ascii=False, indent=2)
            del trainer_state
            gc.collect()

        self.accelerator.wait_for_everyone()

        optimizer = self.optimizer.optimizer if (
            self.accelerator.distributed_type == DistributedType.DEEPSPEED and hasattr(self.optimizer, "optimizer")
        ) else self.optimizer
        optimizer_state = optimizer.state_dict()
        optimizer_state_path = checkpoint_path / f"optimizer_rank_{self.accelerator.process_index:05d}.pt"
        torch.save(optimizer_state, optimizer_state_path)
        del optimizer_state
        gc.collect()

        self._save_lightweight_rng_state(checkpoint_path)

        self.accelerator.wait_for_everyone()

    @staticmethod
    def _save_wan_lora_adapter(bare_model, checkpoint_path: Path) -> None:
        """额外导出独立的 Wan PEFT LoRA adapter，便于轻量分发和复用。"""
        backbone = getattr(bare_model, "backbone", None)
        if not bool(getattr(backbone, "lora_enabled", False)):
            return

        transformer = getattr(backbone, "transformer", None)
        adapter_name = getattr(backbone, "lora_adapter_name", None)
        if transformer is None or not adapter_name:
            raise RuntimeError("Wan LoRA is enabled but its transformer or adapter name is unavailable.")

        try:
            from peft.utils import get_peft_model_state_dict
            from safetensors.torch import save_file
        except ImportError as error:
            raise ImportError(
                "Saving a Wan LoRA adapter requires `peft` and `safetensors`."
            ) from error

        adapter_state = get_peft_model_state_dict(transformer, adapter_name=adapter_name)
        if not adapter_state:
            raise RuntimeError(f"Wan LoRA adapter `{adapter_name}` has no parameters to save.")
        adapter_state = {
            name: tensor.detach().to("cpu", copy=True).contiguous()
            for name, tensor in adapter_state.items()
        }
        adapter_path = checkpoint_path / "wan_lora.safetensors"
        save_file(adapter_state, str(adapter_path))

        peft_config = getattr(transformer, "peft_config", {}).get(adapter_name)
        if peft_config is None:
            raise RuntimeError(f"Wan LoRA adapter `{adapter_name}` has no PEFT config.")
        adapter_metadata = {
            "adapter_name": adapter_name,
            "base_model": getattr(backbone, "model_name", None),
            "target_modules": list(getattr(backbone, "lora_target_modules", ())),
            "peft_config": peft_config.to_dict(),
        }
        with open(checkpoint_path / "wan_lora_config.json", "w", encoding="utf-8") as file:
            json.dump(adapter_metadata, file, ensure_ascii=False, indent=2, default=str)
        logger.info(
            "Saved Wan LoRA adapter: tensors=%d, path=%s",
            len(adapter_state),
            adapter_path,
        )

    def _save_lightweight_checkpoint_metadata(self, checkpoint_path: Path):
        save_lightweight_checkpoint_metadata(
            checkpoint_path,
            self.config,
            local_output_dir=self.local_output_dir,
            network_output_dir=self.network_output_dir,
        )

    @staticmethod
    def _lightweight_scaler_state_path(checkpoint_path: Path) -> Path:
        return checkpoint_path / SCALER_NAME

    def _save_lightweight_scaler_state(self, checkpoint_path: Path):
        save_lightweight_scaler_state(
            checkpoint_path,
            getattr(self.accelerator, "scaler", None),
        )

    def _load_lightweight_scaler_state(self, checkpoint_path: Path):
        load_lightweight_scaler_state(
            checkpoint_path,
            getattr(self.accelerator, "scaler", None),
            logger=logger,
        )

    def _lightweight_rng_state_path(self, checkpoint_path: Path) -> Path:
        return checkpoint_path / f"{RNG_STATE_NAME}_{self.accelerator.process_index}.pkl"

    def _save_lightweight_rng_state(self, checkpoint_path: Path):
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()

        with open(self._lightweight_rng_state_path(checkpoint_path), "wb") as f:
            pickle.dump(rng_state, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_lightweight_rng_state(self, checkpoint_path: Path):
        rng_state_path = self._lightweight_rng_state_path(checkpoint_path)
        if not rng_state_path.exists():
            logger.warning(f"lightweight RNG 状态文件不存在，跳过恢复: {rng_state_path}")
            return

        with open(rng_state_path, "rb") as f:
            rng_state = pickle.load(f)

        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"])
        if torch.cuda.is_available() and "torch_cuda" in rng_state:
            torch.cuda.set_rng_state_all(rng_state["torch_cuda"])

    def _sync_accessed_config_snapshots(self):
        if not isinstance(self.config, AccessTrackedConfig):
            return
        logger.info("📊 Saving accessed configuration...")
        for output_dir in self._all_output_dirs():
            self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
        logger.info("✅ Configuration files saved")

    def _enqueue_checkpoint_sync(self, local_path: Path):
        if self.local_output_dir == self.network_output_dir:
            return

        target_path = self.network_output_dir / local_path.relative_to(self.local_output_dir)
        _launch_background_checkpoint_sync(
            local_path,
            target_path,
            self.sync_log_path,
            cleanup_src=False,
            queue_dir=self.network_output_dir / ".checkpoint_sync_queue",
            keep_local_count=self.local_checkpoint_keep_count,
        )
        logger.info(f"已启动后台同步: {local_path} -> {target_path}")

    def _estimate_checkpoint_bytes(self) -> int:
        candidate_paths = []

        if getattr(self, "resume_from_checkpoint", None):
            candidate_paths.append(Path(self.resume_from_checkpoint))

        local_latest_checkpoint, _ = self._get_latest_checkpoint(str(self.local_checkpoint_dir))
        if local_latest_checkpoint:
            candidate_paths.append(Path(local_latest_checkpoint))

        network_latest_checkpoint, _ = self._get_latest_checkpoint(str(self.network_checkpoint_dir))
        if network_latest_checkpoint:
            candidate_paths.append(Path(network_latest_checkpoint))

        for candidate_path in candidate_paths:
            size_bytes = self._path_size_bytes(candidate_path)
            if size_bytes > 0:
                return size_bytes

        return 12 * 1024 * 1024 * 1024

    def _estimate_retained_checkpoint_bytes(self) -> int:
        keep_existing = max(self.local_checkpoint_keep_count - 1, 0)
        if keep_existing == 0:
            return 0

        complete_entries = _list_complete_checkpoint_entries(self.local_checkpoint_dir)
        retained_entries = complete_entries[-keep_existing:]
        return sum(self._path_size_bytes(entry["path"]) for entry in retained_entries)

    def _ensure_local_checkpoint_capacity(self, checkpoint_path: Path):
        if self.local_output_dir == self.network_output_dir:
            return
        if not self.accelerator.is_main_process:
            return

        required_bytes = int(self._estimate_checkpoint_bytes() * 1.10)
        retained_bytes = self._estimate_retained_checkpoint_bytes()
        available_bytes = shutil.disk_usage(self.local_output_dir).free
        target_exists_bytes = self._path_size_bytes(checkpoint_path)

        if available_bytes + target_exists_bytes < required_bytes + retained_bytes:
            raise RuntimeError(
                "Insufficient local checkpoint space before save: "
                f"available={available_bytes / 1024**3:.2f} GiB, "
                f"required≈{required_bytes / 1024**3:.2f} GiB, "
                f"retained≈{retained_bytes / 1024**3:.2f} GiB, "
                f"checkpoint_path={checkpoint_path}"
            )

    def _maybe_cleanup_local_resume_checkpoint(self):
        global STARTUP_SYNC_INFLIGHT_MARKER

        if self.local_output_dir == self.network_output_dir:
            return
        if self.local_resume_cleanup_started:
            return
        if not self.local_resume_checkpoint_paths_for_cleanup:
            return
        if self.local_checkpoint_keep_count > 1:
            self.local_resume_cleanup_started = True
            return
        if STARTUP_SYNC_INFLIGHT_MARKER is not None and STARTUP_SYNC_INFLIGHT_MARKER.exists():
            logger.info(f"启动阶段本地 checkpoint 补同步尚未完成，暂不清理本地启动 checkpoint: {STARTUP_SYNC_INFLIGHT_MARKER}")
            return

        paths_to_cleanup = [str(path) for path in self.local_resume_checkpoint_paths_for_cleanup if path.exists()]
        if not paths_to_cleanup:
            self.local_resume_cleanup_started = True
            return

        cleanup_script = r"""
import shutil
import sys
from pathlib import Path

for raw_path in sys.argv[1:]:
    path = Path(raw_path)
    if not path.exists():
        continue
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)
"""
        log_file = open(self.sync_log_path, "a")
        subprocess.Popen(
            [sys.executable, "-c", cleanup_script, *paths_to_cleanup],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log_file.close()
        self.local_resume_cleanup_started = True
        if STARTUP_SYNC_INFLIGHT_MARKER is not None:
            _clear_sync_inflight_marker(STARTUP_SYNC_INFLIGHT_MARKER)
            STARTUP_SYNC_INFLIGHT_MARKER = None
        logger.info(f"已启动后台清理本地启动 checkpoint: {paths_to_cleanup}")


def main(cfg) -> None:
    global accelerator
    accelerator = _build_accelerator(cfg)
    accelerator.print(accelerator.state)

    logger.info("VLA Training :: Warming Up")

    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    full_path_dry_run_only = _is_full_path_dry_run(cfg)
    _enforce_launch_guard(cfg, full_path_dry_run_only=full_path_dry_run_only)
    if not full_path_dry_run_only:
        _launch_startup_checkpoint_stage(cfg)
    output_dir = setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    checkpoint_load_summary = _load_full_path_dry_run_checkpoint(cfg, vla)
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    if full_path_dry_run_only:
        batch_summary = None
        forward_summary = None
        if bool(getattr(cfg.trainer, "full_path_dry_run_fetch_batch", False)):
            batch = next(iter(vla_train_dataloader))
            batch_summary = _summarize_batch(batch)
            if bool(getattr(cfg.trainer, "full_path_dry_run_forward_batch", False)):
                vla.eval()
                with torch.no_grad():
                    forward_summary = _run_full_path_dry_run_forward(vla, batch)
        _write_full_path_dry_run_report(
            cfg,
            output_dir=output_dir,
            model=vla,
            dataloader=vla_train_dataloader,
            optimizer=optimizer,
            trainer=trainer,
            batch_summary=batch_summary,
            forward_summary=forward_summary,
            checkpoint_load_summary=checkpoint_load_summary,
        )
        logger.info("MoWA E-001 train_starvla full-path dry-run complete; training skipped.")
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--validate-data-flow",
        action="store_true",
        help="Validate the first MoWA training batches end to end.",
    )
    parser.add_argument(
        "--validation-steps",
        type=int,
        default=2,
        help="Number of initial MoWA batches to validate (default: 2).",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)
    if args.validate_data_flow:
        cfg.framework.mowa.validate_data_flow = True
        cfg.framework.mowa.validation_steps = args.validation_steps

    # Normalise legacy YAML keys into the current `version_id == "0.21"` schema.
    # This is idempotent and does not modify framework class signatures.
    # See bar/config_收紧.md for the rationale.
    cfg = apply_config_compat(cfg)

    # Store source config path for later copying to output dir
    cfg.config_yaml = args.config_yaml

    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
