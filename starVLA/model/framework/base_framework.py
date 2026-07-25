"""
Base framework abstraction providing:
- Pretrained loading (config + normalization stats + weights)
- Action space utilities (dimension, stats, (un)normalization)
- Trainable module discovery helper
Note: No device placement or optimizer concerns handled here (delegated to trainer).
"""

import importlib
import os
import pkgutil
from pathlib import Path
import time
from typing import Any, Dict, List

import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import PretrainedConfig, PreTrainedModel

from starVLA.model.framework.share_tools import (
    dict_to_namespace,
    read_mode_config,
    load_model_weights,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY, FrameworkTools, auto_get_trainable_modules
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)
_FRAMEWORKS_IMPORTED = False


def _auto_import_framework_modules() -> None:
    global _FRAMEWORKS_IMPORTED
    if _FRAMEWORKS_IMPORTED:
        return

    _SKIP = {"__init__", "base_framework", "share_tools"}
    framework_dir = Path(__file__).resolve().parent

    # Scan top-level modules (backwards compat)
    for _, module_name, is_pkg in pkgutil.iter_modules([str(framework_dir)]):
        if module_name in _SKIP:
            continue
        if is_pkg:
            # Scan sub-packages (VLM4A/, WM4A/, etc.)
            sub_dir = framework_dir / module_name
            for _, sub_name, _ in pkgutil.iter_modules([str(sub_dir)]):
                if sub_name.startswith("_"):
                    continue
                importlib.import_module(f"starVLA.model.framework.{module_name}.{sub_name}")
        else:
            importlib.import_module(f"starVLA.model.framework.{module_name}")

    _FRAMEWORKS_IMPORTED = True


def build_framework(cfg): # The single entry point for building different model frameworks
    """
    Build a framework model from config.
    Args:
        cfg: Config object containing `cfg.framework.name`.
    Returns:
        nn.Module: Instantiated framework model.
    """
    if not hasattr(cfg, "framework") or not hasattr(cfg.framework, "name"):
        raise ValueError("Missing `cfg.framework.name`. The framework API now only accepts `framework.name`.")

    _auto_import_framework_modules()

    framework_id = cfg.framework.name
    if framework_id not in FRAMEWORK_REGISTRY._registry:
        available = sorted(FRAMEWORK_REGISTRY._registry.keys())
        raise NotImplementedError(
            f"Framework `{framework_id}` is not implemented. Available frameworks: {available}"
        )

    model_class = FRAMEWORK_REGISTRY[framework_id]
    return model_class(cfg)


# PreTrainedModel, AutoModel, PretrainedConfig,  are so good, find sometime to study them
# TODO @JinhuiYE find sometime to merge yaml config with transformer config


class baseframework(PreTrainedModel):
    """
    Lightweight base class for higher-level VLA model assemblies.
    Subclasses are expected to:
      - Accept a structured config
      - Register components in __init__
      - Use provided helpers for action normalization handling
    """

    def __init__(self, hf_config=PretrainedConfig()) -> None:
        """
        Initialize base nn.Module. Subclasses add components.
        """

        super().__init__(hf_config)

    # ------------------------------------------------------------------
    # Soft-constraint interface: subclasses should override these.
    # Default implementations raise NotImplementedError so that IDE
    # tooling (e.g. pylance, mypy) flags missing overrides, while
    # still allowing PreTrainedModel instantiation (no ABC).
    # ------------------------------------------------------------------

    def forward(self, examples: List[dict], **kwargs) -> dict:
        """Training forward pass.

        Args:
            examples: List[dict], each dict requires at least:
                - image: List[PIL.Image]
                - lang: str
                - action: np.ndarray shaped [T, action_dim]

        Returns:
            dict: Must contain ``"action_loss"`` (torch.Tensor scalar).
                  May contain extra keys for logging (e.g. ``"kl_loss"``).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement forward(examples) -> dict with 'action_loss' key."
        )

    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        """Inference: predict future actions from observations.

        Args:
            examples: Same schema as *forward* (minus ``action`` which is optional).
            **kwargs: Framework-specific inference options (e.g. ``use_ddim``).

        Returns:
            dict: Must contain ``"normalized_actions"`` (np.ndarray [B, T, action_dim]).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement predict_action(examples) -> dict with 'normalized_actions' key."
        )

    # ------------------------------------------------------------------
    # Unified loss interface for Trainer
    # ------------------------------------------------------------------

    def supports_training_tag(self, tag: str) -> bool:
        """Return whether this framework can consume batches for *tag*."""
        if tag == "vla":
            return type(self).forward is not baseframework.forward
        if tag == "vlm":
            return hasattr(self, "qwen_vl_interface") or type(self).forward_vlm is not baseframework.forward_vlm
        return False

    def compute_loss(self, tag: str, batch, loss_scale: dict = None) -> Dict[str, torch.Tensor] | None:
        """Unified forward entry-point: route to the right forward by *tag*.

        The trainer calls ``model.compute_loss(tag, batch)`` for every
        ``(tag, batch)`` pair produced by :class:`DataLoaderManager`.
        The model internally dispatches:

        - ``"vla"`` → ``self.forward(batch)``
        - ``"vlm"`` → ``self.forward_vlm(batch)``

        Subclasses can override this to add more tags (e.g. ``"world"``).

        Args:
            tag: dataset type tag (``"vla"``, ``"vlm"``, …)
            batch: the batch produced by the corresponding DataLoader.
            loss_scale: ``{"vla": 1.0, "vlm": 0.1}`` per-tag loss multiplier.
                        Defaults to 1.0 for unspecified tags.

        Returns:
            dict[str, Tensor] | None: keyed losses (e.g. ``{"action_loss": ...}``).
                Returns ``None`` when this framework does not support the
                incoming dataloader tag so the trainer can ``continue``.
        """
        if not self.supports_training_tag(tag):
            return None

        scale = (loss_scale or {}).get(tag, 1.0)

        if tag == "vla":
            out = self.forward(batch)
        elif tag == "vlm":
            out = self.forward_vlm(batch)
        else:
            return None

        # Apply loss scale and filter to Tensor values only
        return {k: v * scale for k, v in out.items() if isinstance(v, torch.Tensor)}

    def forward_vlm(self, batch) -> Dict[str, torch.Tensor]:
        """VLM forward pass (default implementation).

        Delegates to ``self.qwen_vl_interface(**batch)`` which is present on
        every framework subclass that uses a Qwen VL backbone.

        Subclasses may override to add custom VLM logic.

        Args:
            batch: dict produced by the VLM dataloader.

        Returns:
            dict: Must contain ``"vlm_loss"`` (torch.Tensor scalar).
        """
        if not hasattr(self, "qwen_vl_interface"):
            raise NotImplementedError(
                f"{type(self).__name__} has no `qwen_vl_interface`. "
                "Override forward_vlm() to support VLM training."
            )
        out = self.qwen_vl_interface(**batch)
        return {"vlm_loss": out.loss}

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: str,
        **kwargs,
    ) -> None:
        """
        Restore a model instance from a saved checkpoint.

        Workflow:
            1. Resolve checkpoint path
            2. Load config + dataset normalization statistics
            3. Build model with loaded config
            4. Load state_dict strictly (reports missing/unexpected keys)
            5. Attach normalization stats for later un-normalization

        Args:
            pretrained_checkpoint: Path to .pt file inside run/checkpoints directory.
            **kwargs: Extra constructor overrides passed to subclass.

        Returns:
            baseframework: Instantiated model (left on CPU; caller decides device).

        Raises:
            RuntimeError: If state_dict key mismatch occurs under strict=True.
            FileNotFoundError: If underlying files are missing (surfaced earlier).
        """
        overall_start = time.perf_counter()
        logger.info("[from_pretrained] start ckpt=%s", pretrained_checkpoint)

        stage_start = time.perf_counter()
        model_config, norm_stats = read_mode_config(pretrained_checkpoint)  # read config and norm_stats
        logger.info(
            "[from_pretrained] read_mode_config done in %.2fs",
            time.perf_counter() - stage_start,
        )
        config_overrides = kwargs.pop("config_overrides", None)
        if config_overrides:
            stage_start = time.perf_counter()
            ocfg = OmegaConf.create(model_config)
            override_cfg = OmegaConf.from_dotlist(list(config_overrides))
            ocfg = OmegaConf.merge(ocfg, override_cfg)
            model_config = OmegaConf.to_container(ocfg, resolve=True)
            logger.info(
                "[from_pretrained] applied config_overrides=%s in %.2fs",
                config_overrides,
                time.perf_counter() - stage_start,
            )

        config = dict_to_namespace(model_config)
        model_config = config

        # Save pretrained backbone info before clearing, so we can load it
        # after build_framework (the checkpoint itself omits frozen backbone).
        _pretrained_ckpt = getattr(model_config.trainer, "pretrained_checkpoint", None)
        _reload_modules = getattr(model_config.trainer, "reload_modules", None)
        model_config.trainer.pretrained_checkpoint = None

        stage_start = time.perf_counter()
        FrameworkModel = build_framework(cfg=model_config)
        logger.info(
            "[from_pretrained] build_framework done in %.2fs (%s)",
            time.perf_counter() - stage_start,
            type(FrameworkModel).__name__,
        )
        # set for action un-norm
        FrameworkModel.norm_stats = norm_stats

        # Load pretrained backbone (e.g. OFT-pretrained Wan2.2) on top of
        # the original VLM/WM, so that the following load_model_weights
        # (which omits frozen backbone) lands on the right weights.
        # Aligned with TrainerUtils.load_pretrained_backbones (trainer_tools.py).
        if _pretrained_ckpt and _reload_modules:
            if not os.path.isfile(_pretrained_ckpt):
                raise FileNotFoundError(
                    f"[from_pretrained] pretrained_checkpoint not found: {_pretrained_ckpt}"
                )
            stage_start = time.perf_counter()
            from safetensors.torch import load_file as _load_safe

            _sd = _load_safe(_pretrained_ckpt) if _pretrained_ckpt.endswith(".safetensors") else torch.load(
                _pretrained_ckpt, map_location="cpu", weights_only=True, mmap=True
            )
            for _mod_path in _reload_modules.split(","):
                _mod_path = _mod_path.strip()
                if not _mod_path:
                    continue
                _prefix = _mod_path + "."
                _sub_sd = {k[len(_prefix):]: v for k, v in _sd.items() if k.startswith(_prefix)}
                if not _sub_sd:
                    raise RuntimeError(
                        f"[from_pretrained] no keys matching `{_mod_path}` in {_pretrained_ckpt}"
                    )
                # Navigate to the sub-module (e.g. model.backbone)
                _module = FrameworkModel
                try:
                    for _part in _mod_path.split("."):
                        _module = getattr(_module, _part)
                except AttributeError:
                    raise AttributeError(
                        f"[from_pretrained] cannot find module path `{_mod_path}` in {type(FrameworkModel).__name__}"
                    )
                # Use same loader as training: prefer custom
                # load_pretrained_state_dict (handles LoRA-compatible key remapping).
                _loader = getattr(_module, "load_pretrained_state_dict", None)
                if callable(_loader):
                    _result = _loader(_sub_sd)
                    if _result is not None:
                        _missing = [k for k in (_result.missing_keys or []) if "lora_" not in k]
                        _unexpected = _result.unexpected_keys or []
                        if _missing or _unexpected:
                            logger.warning(
                                "[from_pretrained] backbone `%s` load had mismatches: "
                                "%d missing (non-LoRA), %d unexpected. "
                                "Missing: %s, Unexpected: %s",
                                _mod_path, len(_missing), len(_unexpected),
                                _missing[:5], _unexpected[:5],
                            )
                        else:
                            logger.info(
                                "[from_pretrained] backbone `%s` all keys matched (no missing/unexpected)",
                                _mod_path,
                            )
                else:
                    _module.load_state_dict(_sub_sd, strict=True)
                logger.info(
                    "[from_pretrained] loaded pretrained backbone `%s` from %s in %.2fs",
                    _mod_path, _pretrained_ckpt,
                    time.perf_counter() - stage_start,
                )
            del _sd

        # Load from checkpoint through the shared loader used by training.
        try:
            stage_start = time.perf_counter()
            load_model_weights(FrameworkModel, pretrained_checkpoint, strict=True)
            logger.info(
                "[from_pretrained] load_model_weights done in %.2fs",
                time.perf_counter() - stage_start,
            )
        except RuntimeError as e:
            logger.warning(f"Strict checkpoint load failed for `{pretrained_checkpoint}`: {e}")
            raise

        # **ensure model is on GPU**
        FrameworkModel = FrameworkModel
        logger.info(
            "[from_pretrained] finished in %.2fs",
            time.perf_counter() - overall_start,
        )
        return FrameworkModel
