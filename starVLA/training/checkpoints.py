"""DeepSpeed Universal Checkpoint helpers for StarVLA training.

This module provides thin wrappers around DeepSpeed's universal checkpoint
conversion utilities.  They are only exercised when
trainer.checkpoint_format == "universal" and the training run uses DeepSpeed
ZeRO.  Lightweight / non-DeepSpeed runs do not call these functions.
"""

from __future__ import annotations

from pathlib import Path


REQUIRED_UNIVERSAL_FILES = [
    "zero_to_fp32.py",
]

OPTIONAL_UNIVERSAL_PATTERNS = [
    "*optim_states.pt",
    "*model_states.pt",
    "mp_rank_*_model_states.pt",
]


def is_complete_deepspeed_universal_checkpoint_dir(path: str | Path) -> bool:
    """Return True if ``path`` looks like a DeepSpeed Universal Checkpoint tag.

    A universal checkpoint tag is expected to contain ``zero_to_fp32.py`` and
    at least one model/optimizer state file produced by
    ``deepspeed.checkpoint.ds_to_universal``.
    """
    path = Path(path)
    if not path.is_dir():
        return False

    if not (path / "zero_to_fp32.py").is_file():
        return False

    for pattern in OPTIONAL_UNIVERSAL_PATTERNS:
        if list(path.glob(pattern)):
            return True

    return False


def load_deepspeed_universal_checkpoint(
    model,
    checkpoint_path: str | Path,
    load_optimizer_states: bool = True,
    load_lr_scheduler_states: bool = True,
):
    """Load a DeepSpeed Universal Checkpoint into a DeepSpeed engine.

    The caller must ensure ``model`` is already a DeepSpeed engine.
    """
    try:
        import deepspeed  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "DeepSpeed is required to load a universal checkpoint but is not "
            "available in this environment."
        ) from exc

    checkpoint_path = Path(checkpoint_path)
    if not is_complete_deepspeed_universal_checkpoint_dir(checkpoint_path):
        raise ValueError(
            f"Path does not look like a DeepSpeed Universal Checkpoint: {checkpoint_path}"
        )

    load_path, _ = model.load_checkpoint(
        str(checkpoint_path),
        tag=None,
        load_module_strict=True,
        load_optimizer_states=load_optimizer_states,
        load_lr_scheduler_states=load_lr_scheduler_states,
    )
    return load_path, None


def save_deepspeed_universal_checkpoint(
    checkpoint_path: str | Path,
    save_format: str,
    model,
    accelerator,
    save_lightweight_checkpoint,
    logger,
):
    """Save the current DeepSpeed engine state as a Universal Checkpoint.

    This is a placeholder-compatible implementation.  The full conversion path
    requires a working DeepSpeed ZeRO engine and is intended for multi-GPU
    training; single-card lightweight runs do not exercise this function.
    """
    try:
        import deepspeed  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "DeepSpeed is required to save a universal checkpoint but is not "
            "available in this environment."
        ) from exc

    raise NotImplementedError(
        "save_deepspeed_universal_checkpoint is not implemented for this runtime. "
        "Use trainer.checkpoint_format=deepspeed_state or lightweight instead."
    )
