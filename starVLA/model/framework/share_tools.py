"""
Shared configuration / utility helpers for framework components:
- NamespaceWithGet: lightweight namespace behaving like a dict
- OmegaConf conversion helpers
- Config merging decorator for model __init__
- Checkpoint config/statistics loading
"""

import functools
import gc
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from omegaconf import OmegaConf
import torch
from accelerate.utils.modeling import load_checkpoint_in_model

from starVLA.training.trainer_utils import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


def _resolve_inference_checkpoint_path(pretrained_checkpoint):
    checkpoint_path = Path(pretrained_checkpoint)

    if checkpoint_path.is_file():
        return checkpoint_path, checkpoint_path.parents[1]

    if checkpoint_path.is_dir():
        safetensors_index = checkpoint_path / "model.safetensors.index.json"
        bin_index = checkpoint_path / "pytorch_model.bin.index.json"
        single_file_candidates = (
            checkpoint_path / "model.safetensors",
            checkpoint_path / "pytorch_model.bin",
            checkpoint_path / "pytorch_model.pt",
        )

        if safetensors_index.exists():
            return checkpoint_path, checkpoint_path.parents[1]
        if bin_index.exists():
            return checkpoint_path, checkpoint_path.parents[1]
        for candidate in single_file_candidates:
            if candidate.exists():
                return candidate, checkpoint_path.parents[1]

    overwatch.error(f"❌ Pretrained checkpoint `{pretrained_checkpoint}` does not exist.")
    raise FileNotFoundError(f"Pretrained checkpoint `{pretrained_checkpoint}` does not exist.")


def _resolve_inference_run_files(pretrained_checkpoint):
    checkpoint_pt, run_dir = _resolve_inference_checkpoint_path(pretrained_checkpoint)

    search_dirs = []
    if checkpoint_pt.is_dir():
        search_dirs.append(checkpoint_pt)
    search_dirs.append(run_dir)

    for candidate_dir in search_dirs:
        config_yaml = candidate_dir / "config.yaml"
        dataset_statistics_json = candidate_dir / "dataset_statistics.json"
        if config_yaml.exists() and dataset_statistics_json.exists():
            return checkpoint_pt, candidate_dir, config_yaml, dataset_statistics_json

    raise FileNotFoundError(
        f"Missing `config.yaml` or `dataset_statistics.json` for checkpoint `{pretrained_checkpoint}`. "
        f"Searched in: {[str(path) for path in search_dirs]}"
    )


def _is_safetensors_path(path) -> bool:
    return str(path).endswith(".safetensors")


def _resolve_model_checkpoint_from_dir(path, preferred_format=None):
    checkpoint_path = Path(path)
    if not checkpoint_path.is_dir():
        return None

    def single_file_candidates(fmt):
        if fmt == "safetensors":
            return ("model.safetensors",)
        if fmt == "pt":
            return ("pytorch_model.bin", "pytorch_model.pt")
        return ()

    def index_file_candidate(fmt):
        if fmt == "safetensors":
            return "model.safetensors.index.json"
        if fmt == "pt":
            return "pytorch_model.bin.index.json"
        return None

    def has_complete_index(index_path: Path):
        if not index_path.is_file():
            return False
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            weight_map = payload.get("weight_map", {})
            if not weight_map:
                return False
            shard_names = set(weight_map.values())
            return all((checkpoint_path / shard_name).is_file() for shard_name in shard_names)
        except Exception:
            return False

    formats = []
    if preferred_format in {"pt", "safetensors"}:
        formats.append(preferred_format)
    for fallback_format in ("safetensors", "pt"):
        if fallback_format not in formats:
            formats.append(fallback_format)

    for fmt in formats:
        for candidate_name in single_file_candidates(fmt):
            candidate_path = checkpoint_path / candidate_name
            if candidate_path.is_file():
                return {"path": candidate_path, "format": fmt, "kind": "single_file"}

        index_name = index_file_candidate(fmt)
        if index_name is not None and has_complete_index(checkpoint_path / index_name):
            return {"path": checkpoint_path, "format": fmt, "kind": "sharded_dir"}

    return None


def _resolve_model_checkpoint_artifact(pretrained_checkpoint, preferred_format=None):
    checkpoint_path = Path(pretrained_checkpoint)
    if checkpoint_path.is_dir():
        resolved = _resolve_model_checkpoint_from_dir(checkpoint_path, preferred_format=preferred_format)
        if resolved is None:
            overwatch.error(f"❌ unsupported checkpoint directory format: {pretrained_checkpoint}")
            raise FileNotFoundError(f"Unsupported checkpoint directory format: {pretrained_checkpoint}")
        return resolved

    if checkpoint_path.is_file():
        fmt = "safetensors" if _is_safetensors_path(checkpoint_path) else "pt"
        return {"path": checkpoint_path, "format": fmt, "kind": "single_file"}

    overwatch.error(f"❌ Pretrained checkpoint `{pretrained_checkpoint}` does not exist.")
    raise FileNotFoundError(f"Pretrained checkpoint `{pretrained_checkpoint}` does not exist.")


def load_model_weights(model, pretrained_checkpoint, preferred_format=None, strict=False):
    resolved = _resolve_model_checkpoint_artifact(pretrained_checkpoint, preferred_format=preferred_format)
    checkpoint_path = resolved["path"]

    if resolved["kind"] == "sharded_dir":
        load_checkpoint_in_model(model, str(checkpoint_path), device_map=None, offload_state_dict=True, strict=strict)
        gc.collect()
        return model

    if _is_safetensors_path(checkpoint_path):
        from safetensors.torch import load_file

        checkpoint = load_file(str(checkpoint_path))
    else:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    try:
        model.load_state_dict(checkpoint, strict=strict)
    finally:
        del checkpoint
        gc.collect()
    return model


class NamespaceWithGet(SimpleNamespace):
    def get(self, key, default=None):
        """
        Return attribute value if present, else default (dict-like API).

        Args:
            key: Attribute name.
            default: Fallback if attribute missing.

        Returns:
            Any: Stored value or default.
        """
        return getattr(self, key, default)

    def items(self):
        """
        Iterate (key, value) pairs like dict.items().

        Returns:
            Generator[Tuple[str, Any], None, None]
        """
        return ((key, getattr(self, key)) for key in self.__dict__)

    def __iter__(self):
        """
        Return iterator over attribute keys (enables dict unpacking **obj).

        Returns:
            Iterator[str]
        """
        return iter(self.__dict__)

    def to_dict(self):
        """
        Recursively convert nested NamespaceWithGet objects into plain dicts.

        Returns:
            dict: Fully materialized dictionary structure.
        """
        return {key: value.to_dict() if isinstance(value, NamespaceWithGet) else value for key, value in self.items()}


def dict_to_namespace(d):
    """
    Create an OmegaConf config from a plain dictionary.

    Args:
        d: Input dictionary.

    Returns:
        OmegaConf: DictConfig instance.
    """
    return OmegaConf.create(d)


def _to_omegaconf(x: Any):
    """
    Convert diverse input types into an OmegaConf object.

    Accepted types:
        - None -> empty DictConfig
        - str path -> load YAML/JSON via OmegaConf.load
        - dict -> DictConfig
        - DictConfig / ListConfig -> returned unchanged
        - NamespaceWithGet / SimpleNamespace -> converted via vars()/to_dict()

    Args:
        x: Input candidate.

    Returns:
        OmegaConf: Normalized configuration node.
    """
    if x is None:
        return OmegaConf.create({})
    if isinstance(x, OmegaConf.__class__):  # fallback, typically not hit
        return x
    try:
        # OmegaConf node detection
        from omegaconf import DictConfig, ListConfig

        if isinstance(x, (DictConfig, ListConfig)):
            return x
    except Exception:
        pass

    if isinstance(x, str):
        # treat as path
        return OmegaConf.load(x)
    if isinstance(x, dict):
        return OmegaConf.create(x)
    if isinstance(x, NamespaceWithGet) or isinstance(x, SimpleNamespace):
        # convert to plain dict
        try:
            d = x.to_dict() if hasattr(x, "to_dict") else vars(x)
        except Exception:
            d = vars(x)
        return OmegaConf.create(d)
    # fallback: try to create
    return OmegaConf.create(x)


def merge_pram_config(init):
    """
    Decorator for __init__ to unify config handling.

    Behavior:
        1. Extract 'config' kwarg / arg (path | dict | OmegaConf | namespace)
        2. Convert to OmegaConf
        3. Merge with explicitly passed init parameters (explicit overrides file)
        4. Attach merged config to self.config
        5. Call original __init__ with merged config

    Args:
        init: Original __init__ function.

    Returns:
        Wrapped initializer.
    """

    @functools.wraps(init)
    def wrapper(self, *args, **kwargs):
        # Map positional args to parameter names (excluding self)
        sig = inspect.signature(init)
        param_names = [name for i, (name, p) in enumerate(sig.parameters.items()) if i > 0]

        init_kwargs = {}
        for name, val in zip(param_names, args):
            init_kwargs[name] = val
        # override with explicit kwargs
        init_kwargs.update(kwargs)

        # get provided config (if any)
        provided_config = init_kwargs.get("config", None)

        loaded_cfg = _to_omegaconf(provided_config)

        # build params cfg from explicit init args (other than config)
        params = {k: v for k, v in init_kwargs.items() if k != "config"}
        params_cfg = OmegaConf.create(params) if params else OmegaConf.create({})

        # merge: loaded_cfg <- params_cfg (params override file)
        merged = OmegaConf.merge(loaded_cfg, params_cfg)

        # set on instance
        try:
            # prefer attaching OmegaConf directly
            self.config = merged
        except Exception:
            # fallback to dict
            self.config = OmegaConf.to_container(merged, resolve=True)

        # prepare kwargs for original init: ensure config is the merged OmegaConf
        call_kwargs = dict(init_kwargs)
        call_kwargs["config"] = merged

        # call original __init__ using keyword args only (safer)
        return init(self, **call_kwargs)

    return wrapper


def merge_framework_config(default_config_cls, cfg):
    """
    Merge a framework's default config (dataclass) with the incoming YAML config.

    Rules:
        - default_config_cls provides documented defaults for `cfg.framework`
        - YAML values (cfg.framework) override matching defaults
        - Extra YAML keys not in defaults are preserved (Config-as-API flexibility)
        - Missing YAML keys fall back to defaults (less YAML boilerplate)

    The merge only touches the `cfg.framework` sub-tree; datasets / trainer / etc.
    are left untouched.

    Args:
        default_config_cls: A dataclass **class** (not instance) whose fields() define
                            the default framework config with type hints and comments.
        cfg: The full OmegaConf config (must contain cfg.framework).

    Returns:
        cfg: The same config object with cfg.framework replaced by the merged result.
    """
    import dataclasses

    from omegaconf import DictConfig, OmegaConf

    # 1. Instantiate defaults and convert to OmegaConf
    defaults_instance = default_config_cls()
    defaults_dict = dataclasses.asdict(defaults_instance)
    defaults_omega = OmegaConf.create(defaults_dict)

    # 2. Extract the YAML framework section
    if hasattr(cfg, "framework"):
        # Unwrap AccessTrackedConfig if needed
        yaml_fw = cfg.framework
        if hasattr(yaml_fw, "_cfg"):
            yaml_fw = yaml_fw._cfg
        if not isinstance(yaml_fw, DictConfig):
            yaml_fw = OmegaConf.create(yaml_fw if isinstance(yaml_fw, dict) else {})
    else:
        yaml_fw = OmegaConf.create({})

    # 3. Merge: defaults first, YAML overrides (YAML wins on conflicts)
    merged_fw = OmegaConf.merge(defaults_omega, yaml_fw)

    # 4. Write back into the original cfg
    #    Handle both OmegaConf and AccessTrackedConfig transparently
    if hasattr(cfg, "_cfg") and isinstance(cfg._cfg, DictConfig):
        # AccessTrackedConfig caches child wrappers in _children dict.
        # After replacing the underlying DictConfig node, the old child wrapper
        # still points to the pre-merge node (stale data).  We must invalidate
        # the cache so the next attribute access creates a fresh wrapper around
        # the merged node.
        #
        # However, the old child's _local_accessed set records which keys were
        # already read (e.g. "name" from build_framework).  Deleting the child
        # would lose that tracking info, causing save_accessed_config to omit
        # those keys from config.yaml.  So we preserve and restore it.
        cfg._cfg.framework = merged_fw
        if hasattr(cfg, "_children") and "framework" in cfg._children:
            old_accessed = cfg._children["framework"]._local_accessed.copy()
            del cfg._children["framework"]  # invalidate stale cache
            new_child = cfg.framework  # re-create child around merged_fw
            new_child._local_accessed.update(old_accessed)  # restore tracking
    elif isinstance(cfg, DictConfig):
        cfg.framework = merged_fw
    else:
        # Fallback — try direct attribute setting
        try:
            cfg.framework = merged_fw
        except Exception:
            overwatch.warning("Could not write merged framework config back to cfg.")

    return cfg


def populate_layerwise_dit_cfg(cfg, *, dit_hidden_dim: int, num_dit_layers: int):
    """
    Populate ``framework.action_model.diffusion_model_cfg`` with the DiT shape
    fields required by ``LayerwiseFlowmatchingActionHead``.

    Why this helper exists:
        The action head is intentionally agnostic of the VLM backbone — it only
        consumes ``diffusion_model_cfg``.  Each framework (QwenPI, QwenPI_v3,
        ...) is responsible for deciding the DiT shape (depth + hidden) from
        whatever source it likes (LLM hidden, a compressed projector dim, ...)
        and writing it here BEFORE calling ``get_action_model``.

    Fields written (override any stale YAML values):
        - num_layers           = num_dit_layers
        - input_embedding_dim  = dit_hidden_dim
        - cross_attention_dim  = dit_hidden_dim   (encoder is pre-projected)
        - num_attention_heads  = dit_hidden_dim // attention_head_dim
                                 (uses existing attention_head_dim if set, else 64)

    Args:
        cfg: Full OmegaConf config.
        dit_hidden_dim: DiT internal hidden dim.
        num_dit_layers: Number of DiT cross-attention layers.

    Returns:
        The (mutated) diffusion_model_cfg node.
    """
    dit_cfg = cfg.framework.action_model.diffusion_model_cfg
    head_dim = dit_cfg.get("attention_head_dim", None) or 64
    dit_cfg.attention_head_dim = head_dim
    dit_cfg.num_layers = int(num_dit_layers)
    dit_cfg.input_embedding_dim = int(dit_hidden_dim)
    dit_cfg.cross_attention_dim = int(dit_hidden_dim)
    dit_cfg.num_attention_heads = int(dit_hidden_dim) // int(head_dim)
    return dit_cfg


def read_model_config(pretrained_checkpoint):
    """
    Load global model configuration and dataset normalization statistics
    associated with a saved checkpoint (.pt).

    Expected directory layout:
        <run_dir>/checkpoints/<name>.pt
        <run_dir>/config.json
        <run_dir>/dataset_statistics.json

    Args:
        pretrained_checkpoint: Path to a .pt checkpoint file.

    Returns:
        tuple:
            global_cfg (dict): Loaded config.json contents.
            norm_stats (dict): Dataset statistics for (de)normalization.

    Raises:
        FileNotFoundError: If checkpoint or required JSON files are missing.
        AssertionError: If file suffix or structure invalid.
    """
    checkpoint_pt, run_dir, _, dataset_statistics_json = _resolve_inference_run_files(pretrained_checkpoint)
    overwatch.info(f"Loading from local checkpoint path `{checkpoint_pt}`")

    # Get paths for `config.json`, `dataset_statistics.json` and pretrained checkpoint
    config_json = run_dir / "config.json"
    assert config_json.exists(), f"Missing `config.json` for `{run_dir = }`"

    # Otherwise =>> try looking for a match on `model_id_or_path` on the HF Hub (`model_id_or_path`)
    # Load VLA Config (and corresponding base VLM `ModelConfig`) from `config.json`
    with open(config_json, "r") as f:
        global_cfg = json.load(f)

    # Normalise legacy / pre-v0.21 configs to current schema (idempotent;
    # also ensures `past_action_window_size`, `action_horizon`,
    # `future_action_window_size`, etc. are all present for downstream code).
    try:
        _oc = OmegaConf.create(global_cfg)
        apply_config_compat(_oc)
        global_cfg = OmegaConf.to_container(_oc, resolve=True)
    except Exception as e:
        overwatch.warning(f"apply_config_compat failed on `{config_json}`: {e}")

    # Load Dataset Statistics for Action Denormalization
    with open(dataset_statistics_json, "r") as f:
        norm_stats = json.load(f)
    return global_cfg, norm_stats


def read_mode_config(pretrained_checkpoint):
    """
    Same as read_model_config (legacy duplicate kept for backward compatibility).

    Args:
        pretrained_checkpoint: Path to a .pt checkpoint file.

    Returns:
        tuple:
            vla_cfg (dict)
            norm_stats (dict)
    """
    checkpoint_pt, run_dir, config_yaml, dataset_statistics_json = _resolve_inference_run_files(pretrained_checkpoint)
    overwatch.info(f"Loading from local checkpoint path `{checkpoint_pt}`")

    # Get paths for `config.json`, `dataset_statistics.json` and pretrained checkpoint
    # Otherwise =>> try looking for a match on `model_id_or_path` on the HF Hub (`model_id_or_path`)
    # Load VLA Config (and corresponding base VLM `ModelConfig`) from `config.json`
    try:
        ocfg = OmegaConf.load(str(config_yaml))
        # Normalise legacy / pre-v0.21 configs to current schema (idempotent).
        apply_config_compat(ocfg)
        global_cfg = OmegaConf.to_container(ocfg, resolve=True)
    except Exception as e:
        overwatch.error(f"❌ Failed to load YAML config `{config_yaml}`: {e}")
        raise

    # Load Dataset Statistics for Action Denormalization
    with open(dataset_statistics_json, "r") as f:
        norm_stats = json.load(f)
    return global_cfg, norm_stats


# =============================================================================
# Config compatibility / "tightening" layer (introduced in version_id "0.21").
#
# Goal: keep user-facing YAMLs short and unambiguous while preserving full
# back-compat for old checkpoints' config.yaml. See bar/config_收紧.md for
# the design rationale.
#
# This function is *idempotent* — calling it multiple times yields the same
# result. It does NOT touch framework class signatures; instead it normalises
# the OmegaConf tree so that downstream framework __init__ code (which still
# reads e.g. `future_action_window_size`) keeps working unchanged.
# =============================================================================

CONFIG_VERSION = "0.21"


def apply_config_compat(cfg, *, strict: bool = False):
    """
    Normalise an arbitrary (old or new) starVLA training config into the
    current `version_id == "0.21"` schema.

    Performed transformations (each applied only when needed):

      1.  `framework.action_model.action_horizon` ↔ `future_action_window_size`
          - `action_horizon` is canonical (preferred user-facing name).
          - `future_action_window_size = action_horizon - 1` is auto-filled so
            framework code that still reads the old key keeps working.
          - If both are present and inconsistent, a warning is emitted and
            `action_horizon` wins.

      2.  `framework.action_model.diffusion_model_cfg.output_dim`
          - Auto-filled from `framework.action_model.hidden_size` when missing.

      3.  `framework.action_model.diffusion_model_cfg.cross_attention_dim`
          - Auto-filled from `framework.qwenvl.vl_hidden_dim` when missing.
            Frameworks that further override this at runtime (e.g. QwenGR00T)
            are unaffected.

      4.  `framework.action_model.action_hidden_dim`
          - Auto-filled from `hidden_size` when missing. OFT-family frameworks
            still overwrite this from VLM hidden_size at runtime.

      5.  `framework.action_model.past_action_window_size`
          - Auto-filled to `0` when missing. All released starVLA frameworks
            run with past=0; the field is therefore dropped from user YAMLs
            and only re-materialised here for legacy code that still reads it.

      6.  `cfg.version_id` is stamped to `"0.21"`.

    Args:
        cfg: An OmegaConf DictConfig (or anything _to_omegaconf can wrap).
        strict: If True, raise on inconsistent values instead of warning.

    Returns:
        The same `cfg` object (mutated in place) for chaining convenience.
    """
    from omegaconf import OmegaConf

    if cfg is None:
        return cfg

    src_version = OmegaConf.select(cfg, "version_id", default=None)

    # ---- 1. action_horizon ↔ future_action_window_size ----
    am_path = "framework.action_model"
    am = OmegaConf.select(cfg, am_path, default=None)
    if am is not None:
        ah = OmegaConf.select(am, "action_horizon", default=None)
        fw = OmegaConf.select(am, "future_action_window_size", default=None)

        if ah is None and fw is not None:
            ah = int(fw) + 1
            OmegaConf.update(cfg, f"{am_path}.action_horizon", ah, force_add=True)
        elif ah is not None and fw is None:
            fw = int(ah) - 1
            OmegaConf.update(cfg, f"{am_path}.future_action_window_size", fw, force_add=True)
        elif ah is not None and fw is not None and int(ah) != int(fw) + 1:
            msg = (
                f"[apply_config_compat] inconsistent action_horizon={ah} vs "
                f"future_action_window_size={fw}; expected action_horizon == future + 1. "
                "Using action_horizon as canonical."
            )
            if strict:
                raise ValueError(msg)
            overwatch.warning(msg)
            OmegaConf.update(cfg, f"{am_path}.future_action_window_size", int(ah) - 1, force_add=True)

        # ---- 2 & 3. diffusion_model_cfg auto-fill ----
        dm_path = f"{am_path}.diffusion_model_cfg"
        dm = OmegaConf.select(cfg, dm_path, default=None)
        if dm is not None:
            hidden_size = OmegaConf.select(am, "hidden_size", default=None)
            if OmegaConf.select(dm, "output_dim", default=None) is None and hidden_size is not None:
                OmegaConf.update(cfg, f"{dm_path}.output_dim", int(hidden_size), force_add=True)

            if OmegaConf.select(dm, "cross_attention_dim", default=None) is None:
                vl_hidden = OmegaConf.select(cfg, "framework.qwenvl.vl_hidden_dim", default=None)
                if vl_hidden is not None:
                    OmegaConf.update(cfg, f"{dm_path}.cross_attention_dim", int(vl_hidden), force_add=True)
                # else: leave None — framework __init__ may auto-bind it

        # ---- 4. action_hidden_dim fallback ----
        if OmegaConf.select(am, "action_hidden_dim", default=None) is None:
            hidden_size = OmegaConf.select(am, "hidden_size", default=None)
            if hidden_size is not None:
                OmegaConf.update(cfg, f"{am_path}.action_hidden_dim", int(hidden_size), force_add=True)

        # ---- 5. past_action_window_size default ----
        if OmegaConf.select(am, "past_action_window_size", default=None) is None:
            OmegaConf.update(cfg, f"{am_path}.past_action_window_size", 0, force_add=True)

    # ---- 6. stamp version ----
    if src_version != CONFIG_VERSION:
        try:
            OmegaConf.update(cfg, "version_id", CONFIG_VERSION, force_add=True)
        except Exception:
            try:
                cfg.version_id = CONFIG_VERSION
            except Exception:
                pass
        overwatch.info(f"[apply_config_compat] normalised config from version_id={src_version!r} to {CONFIG_VERSION!r}")

    return cfg


# ──────────────────────────────────────────────────────────────────────
#  Discretised proprioceptive state → instruction prefix (π₀.5 style)
# ──────────────────────────────────────────────────────────────────────
import numpy as _np
from typing import List as _List


def state2str_transform(state: "_np.ndarray", num_bins: int = 256) -> str:
    """Quantise a state vector into ``num_bins`` uniform bins over [-1, 1]
    and return space-separated bin indices.

    Example: [-0.5, 0.1, 0.8] -> "95 133 203"
    """
    discretized_state = _np.digitize(state, bins=_np.linspace(-1, 1, num_bins + 1)[:-1]) - 1
    return " ".join(map(str, discretized_state))


def add_discretized_state_to_instruction(
    instructions: "_List[str]",
    states: "_List[_np.ndarray]",
    num_bins: int = 256,
) -> "_List[str]":
    """Append discretised proprioceptive state tokens to each instruction.

    Format: ``<original instruction> [STATE] <bin indices> [ACTION]``
    Lets the VLM attend to the robot state purely through its existing
    text-token pathway — no extra encoder required (π₀.5 style).
    """
    updated_instructions = []
    for instr, state in zip(instructions, states):
        state_str = state2str_transform(state[0], num_bins=num_bins)
        updated_instructions.append(f"{instr} [STATE] {state_str} [ACTION]")
    return updated_instructions
