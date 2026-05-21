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
import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.checkpointing import load_accelerator_state, load_custom_state
from accelerate.logging import get_logger
from accelerate.utils import DeepSpeedSchedulerWrapper, DistributedType, MODEL_NAME, RNG_STATE_NAME, SAMPLER_NAME, SCALER_NAME, SCHEDULER_NAME
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.dataloader import build_dataloader
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import (
    TrainerUtils,
    build_param_lr_groups,
    setup_optimizer_and_scheduler,
    normalize_dotlist_args,
    _is_complete_deepspeed_checkpoint_dir,
    _is_complete_lightweight_training_checkpoint_dir,
)

deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
accelerator.print(accelerator.state)

STARTUP_CHECKPOINT_STAGE_THREAD = None
STARTUP_CHECKPOINT_STAGE_ERROR = None
STARTUP_LOCAL_AHEAD_OF_NETWORK = None
STARTUP_SYNC_INFLIGHT_MARKER = None

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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


def _iter_model_state_tensors(model):
    for name, param in model.named_parameters():
        yield name, param
    for name, buffer in model.named_buffers():
        yield name, buffer


def _tensor_num_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _streaming_save_model_shards(model, checkpoint_path: Path, save_format: str, max_shard_size) -> None:
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
            save_file(shard_state, str(checkpoint_path / shard_name))
            index_name = "model.safetensors.index.json"
        else:
            shard_name = f"pytorch_model-{shard_idx:05d}.bin"
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
    for name, tensor in _iter_model_state_tensors(bare_model):
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
            if _is_complete_deepspeed_checkpoint_dir(entry) or _is_complete_lightweight_training_checkpoint_dir(entry):
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
            is_complete = _is_complete_deepspeed_checkpoint_dir(entry) or _is_complete_lightweight_training_checkpoint_dir(
                entry
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
            if _is_complete_deepspeed_checkpoint_dir(entry) or _is_complete_lightweight_training_checkpoint_dir(entry):
                checkpoint_entries.append({"path": entry, "step": int(dir_match.group(1)), "complete": True})

    checkpoint_entries.sort(key=lambda x: x["step"])
    return checkpoint_entries


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

    tmp = dst.parent / f".{dst.name}.sync_tmp_{int(time.time() * 1e9)}"
    if src.is_dir():
        shutil.copytree(src, tmp, dirs_exist_ok=True)
    else:
        tmp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, tmp)

    if dst.exists():
        stale = dst.parent / f"{dst.name}.stale_{int(time.time())}"
        dst.rename(stale)

    tmp.rename(dst)

    if cleanup_src and src.exists():
        if src.is_dir():
            shutil.rmtree(src, ignore_errors=True)
        else:
            src.unlink(missing_ok=True)
    elif src.exists() and src.is_dir():
        prune_local_versions(src)

    if marker is not None:
        marker.unlink(missing_ok=True)

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
    dist.barrier()
    return vla_train_dataloader


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
        self.local_checkpoint_keep_count = max(int(getattr(self.config.trainer, "local_checkpoint_keep_count", 1)), 1)

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        _wait_for_startup_checkpoint_stage()
        self._setup_checkpoint_storage()

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
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
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
        self.resume_from_checkpoint = pretrained_checkpoint

        if is_resume:
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                if os.path.isdir(self.resume_from_checkpoint) and _is_complete_deepspeed_checkpoint_dir(
                    self.resume_from_checkpoint
                ):
                    self.resume_requires_training_state = True
                elif os.path.isdir(self.resume_from_checkpoint) and _is_complete_lightweight_training_checkpoint_dir(
                    self.resume_from_checkpoint
                ):
                    self.resume_requires_lightweight_state = True
                    self.model = self.load_pretrained_backbones(
                        self.model, self.resume_from_checkpoint, reload_modules=None
                    )
                else:
                    self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                logger.info(
                    f"Resuming training from checkpoint: {self.resume_from_checkpoint}, steps: {self.completed_steps}"
                )
                return

            logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
            self.completed_steps = 0

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
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
        total_stages = 2 + int(checkpoint_parts["custom_count"] > 0)

        if self.accelerator.is_main_process:
            logger.info(f"目录式 checkpoint 顶层内容: {checkpoint_parts['entries']}")
            logger.info(
                "目录式 checkpoint 检测结果: "
                f"deepspeed_tag={checkpoint_parts['deepspeed_tag']}, "
                f"scheduler_files={checkpoint_parts['scheduler_files']}, "
                f"sampler_files={checkpoint_parts['sampler_files']}, "
                f"rng_files={checkpoint_parts['rng_files']}, "
                f"custom_count={checkpoint_parts['custom_count']}"
            )

        for hook in self.accelerator._load_model_state_pre_hook.values():
            hook([], str(checkpoint_path))

        logger.info(f"[1/{total_stages}] 开始加载 DeepSpeed 模型/优化器状态: {checkpoint_path}")
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

        if checkpoint_parts["custom_count"] > 0:
            logger.info(f"[3/{total_stages}] 开始加载 {checkpoint_parts['custom_count']} 个自定义状态")
            for index, obj in enumerate(self.accelerator._custom_objects):
                load_custom_state(obj, str(checkpoint_path), index)
            logger.info(f"[3/{total_stages}] 自定义状态加载完成")

        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _load_lightweight_training_state(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        total_stages = 2

        logger.info(f"[1/{total_stages}] 开始加载 lightweight optimizer 状态: {checkpoint_path / 'optimizer.pt'}")
        optimizer_state = torch.load(
            checkpoint_path / "optimizer.pt",
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED and hasattr(self.optimizer, "optimizer"):
            self.optimizer.optimizer.load_state_dict(
                [optimizer_state],
                load_optimizer_states=True,
                load_from_fp32_weights=False,
            )
        else:
            self.optimizer.load_state_dict(optimizer_state)
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

        with open(checkpoint_path / "trainer_state.json", "r", encoding="utf-8") as f:
            trainer_state = json.load(f)
        self.completed_steps = int(trainer_state.get("completed_steps", self.completed_steps))
        del trainer_state
        gc.collect()
        logger.info(f"[2/{total_stages}] lightweight scheduler / trainer 状态加载完成")

        self.accelerator.print(f"Resumed lightweight training state from checkpoint: {checkpoint_path}")

    @staticmethod
    def _inspect_directory_checkpoint(checkpoint_path: Path) -> dict:
        entries = sorted(path.name for path in checkpoint_path.iterdir())
        latest_file = checkpoint_path / "latest"
        deepspeed_tag = MODEL_NAME
        if latest_file.exists():
            deepspeed_tag = latest_file.read_text().strip() or MODEL_NAME

        scheduler_files = [name for name in entries if name.startswith(SCHEDULER_NAME)]
        sampler_files = [name for name in entries if name.startswith(SAMPLER_NAME) or name.startswith("dl_state_dict")]
        rng_files = [name for name in entries if name.startswith(RNG_STATE_NAME)]
        custom_files = [name for name in entries if name.startswith("custom_checkpoint_")]

        return {
            "entries": entries,
            "deepspeed_tag": deepspeed_tag,
            "scheduler_files": scheduler_files,
            "sampler_files": sampler_files,
            "rng_files": rng_files,
            "custom_count": len(custom_files),
        }

    def _save_checkpoint(self):
        """Save current training state."""
        save_format = getattr(self.config.trainer, "save_format", "safetensors")
        checkpoint_path = self.local_checkpoint_dir / f"steps_{self.completed_steps}"
        self._ensure_local_checkpoint_capacity(checkpoint_path)

        if self.accelerator.distributed_type == DistributedType.DEEPSPEED and self.save_with_training_state:
            self.accelerator.save_state(output_dir=str(checkpoint_path), safe_serialization=(save_format == "safetensors"))
        elif self.accelerator.is_main_process and self.save_checkpoint_as_directory:
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

        if self.accelerator.is_main_process:
            self._append_summary_entry({"steps": self.completed_steps})
            self._sync_accessed_config_snapshots()
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")
            self._enqueue_checkpoint_sync(checkpoint_path)

        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """Record training metrics."""
        if self.completed_steps % self.config.trainer.logging_frequency == 0 and dist.get_rank() == 0:
            last_lrs = self.lr_scheduler.get_last_lr()
            for i, group in enumerate(self.optimizer.param_groups):
                group_name = group.get("name", str(i))
                metrics[f"learning_rate/{group_name}"] = last_lrs[i] if i < len(last_lrs) else last_lrs[-1]
            metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)
            wandb.log(metrics, step=self.completed_steps)
            logger.info(f"Step {self.completed_steps}, Loss: {metrics})")
            self._maybe_cleanup_local_resume_checkpoint()

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

            if self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics)

            step_metrics["timing/data"] = t_end_data - t_start_data
            step_metrics["timing/model"] = t_end_model - t_start_model
            self._log_metrics(step_metrics)

            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """Run simple action-eval on current batch and attach score to metrics."""
        examples = self._get_next_batch()
        actions = [example["action"] for example in examples]
        output_dict = self.accelerator.unwrap_model(self.model).predict_action(
            examples=examples, use_ddim=True, num_ddim_steps=20
        )

        if self.accelerator.is_main_process:
            normalized_actions = output_dict["normalized_actions"]
            actions = np.array(actions)
            num_pots = np.prod(actions.shape)
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            step_metrics["mse_score"] = score / num_pots

        del examples
        if dist.is_initialized():
            dist.barrier()
        return step_metrics

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_vla, batch_vlm=None):
        """Execute single training step."""
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
                total_loss = action_loss

            self.accelerator.backward(total_loss)

            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            self.optimizer.step()
            # Only step the LR scheduler when gradients are actually synced
            # (i.e., not mid-accumulation). Without this guard the scheduler
            # runs gradient_accumulation_steps times faster than intended,
            # causing warmup to end too early and cosine decay to bottom out
            # at min_lr well before max_train_steps is reached.
            if self.accelerator.sync_gradients:
                self.lr_scheduler.step()

        return {
            "action_dit_loss": action_loss.item(),
        }

    def _finalize_training(self):
        """Training end processing."""
        save_format = getattr(self.config.trainer, "save_format", "safetensors")
        final_checkpoint = self.local_output_dir / "final_model"
        self._ensure_local_checkpoint_capacity(final_checkpoint)
        os.makedirs(final_checkpoint, exist_ok=True)

        if self.accelerator.distributed_type == DistributedType.DEEPSPEED and self.save_with_training_state:
            self.accelerator.save_state(output_dir=str(final_checkpoint), safe_serialization=(save_format == "safetensors"))
        elif self.accelerator.is_main_process and self.save_checkpoint_as_directory:
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

        if self.accelerator.is_main_process:
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")
            self._enqueue_checkpoint_sync(final_checkpoint)

        if self.accelerator.is_main_process:
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
        bare_model = self.accelerator.unwrap_model(self.model)
        _streaming_save_model_shards(bare_model, checkpoint_path, save_format, self.checkpoint_max_shard_size)
        gc.collect()
        optimizer_state = self.optimizer.state_dict()
        torch.save(optimizer_state, checkpoint_path / "optimizer.pt")
        del optimizer_state
        gc.collect()

        scheduler_state = self.lr_scheduler.state_dict()
        torch.save(scheduler_state, checkpoint_path / "scheduler.pt")
        del scheduler_state
        gc.collect()

        trainer_state = {
            "completed_steps": self.completed_steps,
            "save_format": save_format,
            "checkpoint_type": "lightweight_training",
        }
        with open(checkpoint_path / "trainer_state.json", "w", encoding="utf-8") as f:
            json.dump(trainer_state, f, ensure_ascii=False, indent=2)
        del trainer_state
        gc.collect()

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
    logger.info("VLA Training :: Warming Up")

    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    output_dir = setup_directories(cfg=cfg)
    _launch_startup_checkpoint_stage(cfg)
    vla = build_framework(cfg)
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
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

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
