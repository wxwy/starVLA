"""
metrics.py

Utility classes defining a Metrics container and multiple Trackers to enable model/stage-specific logging to various
endpoints (e.g., JSONL local logs, Weights & Biases).
"""

from typing import Tuple
import re
import json
import gc
import shutil
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from transformers import get_scheduler
from accelerate.utils import SCALER_NAME
from omegaconf import OmegaConf

from accelerate.logging import get_logger
from starVLA.model.framework.share_tools import (
    _is_safetensors_path as _shared_is_safetensors_path,
    _resolve_model_checkpoint_from_dir as _shared_resolve_model_checkpoint_from_dir,
    load_model_weights as _shared_load_model_weights,
)
from starVLA.model.modules.starflow_vla.mapping import save_starflow_checkpoint_mapping
from starVLA.training.checkpoints import is_complete_deepspeed_universal_checkpoint_dir
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig

logger = get_logger(__name__)


# === Define Tracker Interface ===
#

# utils/cli_parser.py


def normalize_dotlist_args(args):
    """
    Convert ['--x.y', 'val'] and ['--flag'] → ['x.y=val', 'flag=true']
    """
    normalized = []
    skip = False
    for i in range(len(args)):
        if skip:
            skip = False
            continue

        arg = args[i]
        if arg.startswith("--"):
            key = arg.lstrip("-")
            if "=" in key:
                normalized.append(key)
            elif i + 1 < len(args) and not args[i + 1].startswith("--"):
                normalized.append(f"{key}={args[i + 1]}")
                skip = True
            else:
                normalized.append(f"{key}=true")
        else:
            pass  # skip orphaned values
    return normalized


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Build AdamW optimizer + LR scheduler from cfg.trainer.

    Supports:
    - Per-module learning rates via cfg.trainer.learning_rate
    - cosine_with_min_lr (or any transformers scheduler) via cfg.trainer.lr_scheduler_type
    """
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    fused_available = torch.cuda.is_available()
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
        fused=fused_available,
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # Strip keys unknown to transformers' get_scheduler before passing kwargs.
    sched_kwargs = cfg.trainer.scheduler_specific_kwargs
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=sched_kwargs,
    )

    return optimizer, lr_scheduler


def build_param_lr_groups(model, cfg):
    """
    build multiple param groups based on cfg.trainer.learning_rate.
    support specifying different learning rates for different modules, the rest use base.

    Args:
        vla: nn.Module model object
        cfg: config object, requires cfg.trainer.learning_rate dictionary

    Returns:
        List[Dict]: param_groups that can be used to build optimizer with torch.optim
    """

    lr_cfg = cfg.trainer.learning_rate
    base_lr = lr_cfg.get("base", 1e-4)  # default base learning rate

    freeze_modules = cfg.trainer.get("freeze_modules", "")
    if not isinstance(freeze_modules, str):
        freeze_modules = ""
    freeze_patterns = [p.strip() for p in freeze_modules.split(",") if p.strip()]

    used_params = set()
    frozen_params = set()
    param_groups = []

    for freeze_path in freeze_patterns:
        module = model
        try:
            for attr in freeze_path.split("."):
                module = getattr(module, attr)
            frozen_params.update(id(p) for p in module.parameters())
        except AttributeError:
            print(f"⚠️ freeze module path does not exist: {freeze_path}")
            continue

    for module_name, lr in lr_cfg.items():
        if module_name == "base":
            continue
        # try to find the module under vla by module_name (support nested paths)
        module = model
        try:
            for attr in module_name.split("."):
                module = getattr(module, attr)
            # filter out frozen parameters
            params = [p for p in module.parameters() if id(p) not in frozen_params]
            if params:  # only add param group if there are trainable parameters
                param_groups.append({"params": params, "lr": lr, "name": module_name})
                used_params.update(id(p) for p in params)
        except AttributeError:
            ReferenceError(f"⚠️ module path `{module_name}` not found in vla")

    # assign base learning rate to the remaining unused parameters (exclude frozen ones)
    other_params = [p for p in model.parameters() if id(p) not in used_params and id(p) not in frozen_params]
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr, "name": "base"})

    return param_groups


import torch.distributed as dist


def only_main_process(func):
    """
    decorator: only run in main process (rank=0)
    """

    def wrapper(*args, **kwargs):
        if dist.is_initialized() and dist.get_rank() != 0:
            return None  # non-main process does not execute
        return func(*args, **kwargs)

    return wrapper


from torchvision.ops import box_iou
from PIL import Image


def resize_images(images, target_size=(224, 224)):
    """
    recursively resize all images in the nested list.

    :param images: nested list of images or single image.
    :param target_size: target size (width, height) after resizing.
    :return: resized images list, keeping the original nested structure.
    """
    if isinstance(images, Image.Image):  # if it is a single PIL image
        return images.resize(target_size)
    elif isinstance(images, list):  # if it is a list, recursively process each element
        return [resize_images(img, target_size) for img in images]
    else:
        raise ValueError("Unsupported image type or structure.")


import torch.distributed as dist


class TrainerUtils:
    @staticmethod
    def freeze_backbones(model, freeze_modules=""):
        """
        directly freeze the specified submodules based on the relative module path list (patterns), no longer recursively find all submodule names:
          - patterns: read from config.trainer.freeze_modules, separated by commas to get the "relative path" list
            for example "qwen_vl_interface, action_model.net",
            it means to freeze model.qwen_vl_interface and model.action_model.net.

        Args:
            model: nn.Module model object
            freeze_modules: relative module path list (patterns)

        Returns:
            model: nn.Module model object
        return:
          - model:
        """
        frozen = []
        print("#"*30)
        print(freeze_modules)
        if freeze_modules and type(freeze_modules) == str:
            # split and remove whitespace
            patterns = [p.strip() for p in freeze_modules.split(",") if p.strip()] if freeze_modules else []

            for path in patterns:
                # split the "relative path" by dots, for example "action_model.net" → ["action_model", "net"]
                attrs = path.split(".")
                module = model
                try:
                    for attr in attrs:
                        module = getattr(module, attr)
                    # if the module is successfully get, freeze it and its all submodule parameters
                    for param in module.parameters():
                        param.requires_grad = False
                    frozen.append(path)
                except AttributeError:
                    # if the attribute does not exist, skip and print warning
                    print(f"⚠️ module path does not exist, cannot freeze: {path}")
                    continue

        # accelerator.wait_for_everyone()  # synchronize when distributed training
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            print(f"🔒 Frozen modules with re pattern: {frozen}")
        return model

    @staticmethod
    def print_trainable_parameters(model):
        """
        print the total number of parameters and trainable parameters of the model
        :param model: PyTorch model instance
        """
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        print("📊 model parameter statistics:")
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"# Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable"
        )
        return num_params, num_trainable_params

    @staticmethod
    def load_pretrained_backbones(model, checkpoint_path=None, reload_modules=None, preferred_format=None):
        """
        load checkpoint:
        - if reload_modules is set, load by path part
        - otherwise → load the entire model parameters (overwrite model)

        return:
            replace, loaded_modules: list of module paths that successfully loaded parameters; if global load, then ["<full_model>"]
        """
        if not checkpoint_path:
            return []
        if os.path.isdir(checkpoint_path):
            resolved_checkpoint = _resolve_model_checkpoint_from_dir(checkpoint_path, preferred_format=preferred_format)
            if resolved_checkpoint is None:
                raise RuntimeError(f"❌ unsupported checkpoint directory format: {checkpoint_path}")
            if reload_modules:
                if resolved_checkpoint["kind"] != "single_file":
                    raise RuntimeError(f"❌ partial module reload does not support sharded checkpoint directory: {checkpoint_path}")
                checkpoint_path = resolved_checkpoint["path"]
            else:
                try:
                    _shared_load_model_weights(model, checkpoint_path, preferred_format=preferred_format, strict=False)
                    if (not dist.is_initialized()) or dist.get_rank() == 0:
                        print("✅ loaded <full_model> model parameters")
                    gc.collect()
                    return model
                except Exception as e:
                    raise RuntimeError(f"❌ loading sharded checkpoint failed: {e}")
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            print(f"📦 loading checkpoint: {checkpoint_path}")
        try:
            if _is_safetensors_path(checkpoint_path):
                from safetensors.torch import load_file

                checkpoint = load_file(checkpoint_path)
            else:
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=True,
                    mmap=True,
                )
        except Exception as e:
            raise RuntimeError(f"❌ loading checkpoint failed: {e}")

        loaded_modules = []

        if reload_modules:  # partial load
            module_paths = [p.strip() for p in reload_modules.split(",") if p.strip()]
            for path in module_paths:
                reload_modules = path.split(".")
                module = model
                try:
                    for module_name in reload_modules:  # find the module to modify level by level
                        module = getattr(module, module_name)
                    prefix = path + "."
                    sub_state_dict = {k[len(prefix) :]: v for k, v in checkpoint.items() if k.startswith(prefix)}
                    if sub_state_dict:
                        module.load_state_dict(sub_state_dict, strict=True)
                        if (not dist.is_initialized()) or dist.get_rank() == 0:
                            print(f"✅ parameters loaded to module '{path}'")
                        loaded_modules.append(path)
                    else:
                        print(f"⚠️ parameters not found in checkpoint '{path}'")
                except AttributeError:
                    print(f"❌ cannot find module path: {path}")
        else:  # full load
            try:
                model.load_state_dict(checkpoint, strict=False)
                if (not dist.is_initialized()) or dist.get_rank() == 0:
                    print("✅ loaded <full_model> model parameters")
                loaded_modules = ["<full_model>"]
            except Exception as e:
                raise RuntimeError(f"❌ loading full model failed: {e}")
            finally:
                del checkpoint
                gc.collect()
        return model

    @staticmethod
    def print_freeze_status(model):
        """
        print the freezing status of each parameter in the model
        :param model: PyTorch model instance
        """
        for name, param in model.named_parameters():
            status = "Frozen" if not param.requires_grad else "Trainable"
            print(f"{name:60s}  |  {status}")

    @staticmethod
    def setup_distributed_training(accelerator, *components):
        """
        use Accelerator to prepare distributed training components
        :param accelerator: Accelerate instance
        :param components: any number of components (such as model, optimizer, dataloader, etc.)
        :return: prepared distributed components (in the same order as input)
        """

        # use accelerator.prepare method to wrap components
        prepared_components = accelerator.prepare(*components)
        return prepared_components

    @staticmethod
    def euclidean_distance(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        return np.linalg.norm(predicted - ground_truth)

    @staticmethod
    def _reset_dataloader(dataloader, epoch_counter):
        """safe reset dataloader iterator"""
        # 1. update epoch counter
        epoch_counter += 1

        # 2. set new epoch (distributed core)
        if hasattr(dataloader, "sampler") and callable(getattr(dataloader.sampler, "set_epoch", None)):
            dataloader.sampler.set_epoch(epoch_counter)

        # 3. create new iterator
        return iter(dataloader), epoch_counter

    @staticmethod
    def compute_grad_angle_with_stats(grads_a: list[torch.Tensor], grads_v: list[torch.Tensor]) -> Tuple[float, float]:
        """
        compute the cosine angle between two groups of gradient vectors (degrees), and calculate the average angle and variance.
        grads_a, grads_v: gradient Tensor list corresponding to the same parameter list interface_params
        return:
            mean_angle_deg: average angle (degrees)
            angle_variance: angle variance
        """
        angle_degs = []

        # compute the cosine angle between each gradient block grads_a[0].shape = 1280, 3, 14, 14
        # grads_1 = grads_a[0][0]  # [3, 14, 14]
        # grads_2 = grads_v[0][0]
        # grads_a = grads_1.view(-1, 3)  # reshape to [196, 3]
        # grads_v = grads_2.view(-1, 3)

        # lang linear
        # reshape to 14*14, 3
        # layer
        grads_action = grads_a[0]  # [2048, 11008]
        grads_action = grads_action[
            :32, :7
        ]  # only take the first 7 elements, avoid cosim failure in high-dimensional space
        grads_vl = grads_v[0]  # [2048, 11008]
        grads_vl = grads_vl[
            :32, :7
        ]  # only take the first 32 elements, 7 dimensions, avoid cosim failure in high-dimensional space
        for g_a, g_v in zip(grads_action, grads_vl):
            dot = torch.sum(g_a * g_v)
            norm_a_sq = torch.sum(g_a * g_a)
            norm_v_sq = torch.sum(g_v * g_v)

            # avoid division by zero
            norm_a = torch.sqrt(norm_a_sq + 1e-16)
            norm_v = torch.sqrt(norm_v_sq + 1e-16)

            cos_sim = (dot / (norm_a * norm_v)).clamp(-1.0, 1.0)
            angle_rad = torch.acos(cos_sim)
            angle_deg = angle_rad * (180.0 / torch.pi)

            angle_degs.append(angle_deg.item())

        # compute the average angle and variance
        angle_degs_tensor = torch.tensor(angle_degs)
        mean_angle_deg = torch.mean(angle_degs_tensor).item()
        angle_variance = torch.sqrt(torch.var(angle_degs_tensor)).item()
        # accelerator.wait_for_everyone()
        return mean_angle_deg, angle_variance

    @staticmethod
    def pcgrad_project(grads_a: list[torch.Tensor], grads_v: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        apply PCGrad projection to the second group of gradients grads_v, suppress negative transfer between grads_a and grads_v
        if the dot product of two groups of gradients < 0, then:
            grads_v <- grads_v - (dot / ||grads_a||^2) * grads_a
        return the new grads_v list
        """
        # first compute dot and ||grads_a||^2
        dot, norm_a_sq = 0.0, 0.0
        for g_a, g_v in zip(grads_a, grads_v):
            dot += torch.sum(g_a * g_v)
            norm_a_sq += torch.sum(g_a * g_a)

        if dot < 0:
            coeff = dot / (norm_a_sq + 1e-6)
            # projection
            grads_v = [g_v - coeff * g_a for g_a, g_v in zip(grads_a, grads_v)]

        return grads_v

    @staticmethod
    def eval_qwenpi(qwenpi, dataloader, num_batches=20):
        """
        evaluate QwenQFormerDiT model, compute IoU and action distance.

        Args:
            qwenpi: QwenQFormerDiT model instance.
            dataloader: data loader.
            num_batches: number of batches to evaluate.

        Returns:
            dict: contains IoU and action distance evaluation results.
        """
        iou_scores = []
        action_distances = []
        count = 0

        dataset_iter = iter(dataloader)
        while count < num_batches:
            try:
                batch_samples = next(dataset_iter)
                count += 1
            except StopIteration:
                break

            # extract data
            images = [example["image"] for example in batch_samples]
            instructions = [example["lang"] for example in batch_samples]
            actions = [example["action"] for example in batch_samples]
            solutions = [example["solution"] for example in batch_samples]

            # model prediction
            predicted_solutions, normalized_actions = qwenpi.predict_action_withCoT(
                images=images, instructions=instructions, use_ddim=False, num_ddim_steps=20
            )

            # extract and convert predicted results
            parsed_solutions = []
            for solution in predicted_solutions:
                parsed_solution = TrainerUtils.extract_json_from_string(solution)
                parsed_solutions.append(parsed_solution)

            # compute IoU
            for pred_dict, gt_dict in zip(parsed_solutions, solutions):
                pred_pick_bbox = torch.tensor(pred_dict["pick"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)
                gt_pick_bbox = torch.tensor(gt_dict["pick"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)
                pred_place_bbox = torch.tensor(pred_dict["place"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)
                gt_place_bbox = torch.tensor(gt_dict["place"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)

                pick_iou = box_iou(pred_pick_bbox, gt_pick_bbox).item()
                place_iou = box_iou(pred_place_bbox, gt_place_bbox).item()

                iou_scores.append({"pick_iou": pick_iou, "place_iou": place_iou})

            # compute action distance
            actions = np.array(actions)  # convert to numpy array
            num_pots = np.prod(actions.shape)  # B*len*dim
            action_distance = TrainerUtils.euclidean_distance(normalized_actions, actions)
            average_action_distance = action_distance / num_pots
            action_distances.append(average_action_distance)

        # summarize results
        avg_action_distance = np.mean(action_distances)
        return {"iou_scores": iou_scores, "average_action_distance": avg_action_distance}

    @staticmethod
    def extract_json_from_string(input_string):
        """
        extract valid JSON part from string and convert to dictionary.

        Args:
            input_string (str): string containing extra characters.

        Returns:
            dict: dictionary extracted and parsed.
        """
        json_match = re.search(r"{.*}", input_string, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"JSON decode failed: {e}")
                return None
        else:
            print("No valid JSON part found")
            return None

    def _get_latest_checkpoint(self, checkpoint_dir):
        """Find the latest checkpoint in the directory based on step number."""
        if not os.path.exists(checkpoint_dir):
            self.accelerator.print(f"No checkpoint directory found at {checkpoint_dir}")
            return None, 0

        checkpoint_entries = []
        for entry in os.listdir(checkpoint_dir):
            entry_path = os.path.join(checkpoint_dir, entry)
            file_match = re.match(r"steps_(\d+)_(?:pytorch_model\.pt|model\.safetensors)$", entry)
            dir_match = re.match(r"steps_(\d+)$", entry)

            if file_match and os.path.isfile(entry_path):
                checkpoint_entries.append((entry, int(file_match.group(1))))
            elif dir_match and os.path.isdir(entry_path):
                if _is_complete_deepspeed_checkpoint_dir(entry_path):
                    checkpoint_entries.append((entry, int(dir_match.group(1))))
                elif _is_complete_deepspeed_universal_checkpoint_dir(entry_path):
                    checkpoint_entries.append((entry, int(dir_match.group(1))))
                elif _is_complete_lightweight_training_checkpoint_dir(entry_path):
                    checkpoint_entries.append((entry, int(dir_match.group(1))))
                else:
                    self.accelerator.print(f"Skipping incomplete checkpoint directory: {entry_path}")

        if not checkpoint_entries:
            self.accelerator.print(f"No checkpoints found in {checkpoint_dir}")
            return None, 0

        # Sort by step number and get the latest checkpoint
        checkpoint_entries.sort(key=lambda x: x[1])
        latest_checkpoint, completed_steps = checkpoint_entries[-1]

        latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        self.accelerator.print(f"Latest checkpoint found: {latest_checkpoint_path}")
        return latest_checkpoint_path, completed_steps

import os


def is_main_process():
    rank = int(os.environ.get("RANK", 0))  # if RANK is not set, default to 0
    return rank == 0


def _is_safetensors_path(path):
    """Check if a path refers to a safetensors file."""
    return _shared_is_safetensors_path(path)


def _resolve_model_checkpoint_from_dir(path, preferred_format=None):
    """Resolve checkpoint artifacts through the shared framework loader helper."""
    return _shared_resolve_model_checkpoint_from_dir(path, preferred_format=preferred_format)


def _get_model_checkpoint_file_from_dir(path, preferred_format=None):
    """Resolve the model weight file from a lightweight checkpoint directory."""
    resolved = _resolve_model_checkpoint_from_dir(path, preferred_format=preferred_format)
    if resolved is None or resolved["kind"] != "single_file":
        return None
    return resolved["path"]


def _is_complete_model_checkpoint_dir(path):
    """Check whether a lightweight model checkpoint directory contains model weights or a shard index."""
    return _resolve_model_checkpoint_from_dir(path) is not None


def _is_complete_lightweight_training_checkpoint_dir(path):
    """Check whether a lightweight training checkpoint contains model weights plus optimizer/scheduler state."""
    required_files = (
        os.path.join(path, "scheduler.pt"),
        os.path.join(path, "trainer_state.json"),
    )
    has_optimizer = os.path.isfile(os.path.join(path, "optimizer.pt")) or any(
        re.match(r"optimizer_rank_\d+\.pt$", name) for name in os.listdir(path)
    )
    return (
        _is_complete_model_checkpoint_dir(path)
        and has_optimizer
        and all(os.path.isfile(file_path) for file_path in required_files)
    )


def _lightweight_scaler_state_path(checkpoint_path: str | Path) -> Path:
    return Path(checkpoint_path) / SCALER_NAME


def save_lightweight_scaler_state(checkpoint_path: str | Path, scaler) -> Path:
    checkpoint_path = Path(checkpoint_path)
    scaler_payload = {
        "has_scaler": scaler is not None,
        "state_dict": scaler.state_dict() if scaler is not None else None,
    }
    output_path = _lightweight_scaler_state_path(checkpoint_path)
    torch.save(scaler_payload, output_path)
    return output_path


def load_lightweight_scaler_state(checkpoint_path: str | Path, scaler, logger=None) -> bool:
    scaler_state_path = _lightweight_scaler_state_path(checkpoint_path)
    if not scaler_state_path.exists():
        if logger is not None:
            logger.warning(f"lightweight scaler 状态文件不存在，跳过恢复: {scaler_state_path}")
        return False

    scaler_payload = torch.load(
        scaler_state_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )

    if isinstance(scaler_payload, dict) and "has_scaler" in scaler_payload:
        if not scaler_payload["has_scaler"]:
            if logger is not None:
                logger.info("checkpoint 记录当前训练未启用 scaler，跳过恢复。")
            return False
        scaler_state = scaler_payload.get("state_dict")
    else:
        scaler_state = scaler_payload

    if scaler is None:
        if logger is not None:
            logger.warning("当前 accelerator 未启用 scaler，跳过恢复 checkpoint 中的 scaler 状态。")
        return False
    if scaler_state is None:
        if logger is not None:
            logger.warning("checkpoint scaler 状态为空，跳过恢复。")
        return False

    scaler.load_state_dict(scaler_state)
    return True


def save_lightweight_checkpoint_metadata(
    checkpoint_path: str | Path,
    config,
    *,
    local_output_dir: str | Path,
    network_output_dir: str | Path,
):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    if isinstance(config, AccessTrackedConfig):
        config.save_accessed_config(checkpoint_path / "config.yaml", use_original_values=False)
        full_cfg = config.unwrap()
    else:
        OmegaConf.save(config, checkpoint_path / "config.yaml", resolve=True)
        full_cfg = config

    OmegaConf.save(full_cfg, checkpoint_path / "config.full.yaml", resolve=True)

    for source_dir in (Path(local_output_dir), Path(network_output_dir)):
        dataset_statistics_path = source_dir / "dataset_statistics.json"
        if dataset_statistics_path.exists():
            shutil.copy2(dataset_statistics_path, checkpoint_path / "dataset_statistics.json")
            break

    save_starflow_checkpoint_mapping(checkpoint_path, config)


def _is_complete_deepspeed_checkpoint_dir(path):
    """Check whether a DeepSpeed checkpoint directory contains the minimum files required for resume-training."""
    if not os.path.isdir(path):
        return False

    latest_file = os.path.join(path, "latest")
    model_file = os.path.join(path, "pytorch_model", "mp_rank_00_model_states.pt")
    optim_file = os.path.join(path, "pytorch_model", "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt")
    rng_file = os.path.join(path, "random_states_0.pkl")

    required_files = [latest_file, model_file, optim_file, rng_file]
    return all(os.path.isfile(file_path) for file_path in required_files)


def _is_complete_deepspeed_universal_checkpoint_dir(path):
    """Check whether a DeepSpeed Universal checkpoint tag directory is complete enough for resume."""
    return is_complete_deepspeed_universal_checkpoint_dir(path)
