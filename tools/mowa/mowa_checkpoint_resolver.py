"""MoWA checkpoint reference resolver for smoke/eval tools."""

from __future__ import annotations

import json
from pathlib import Path


LATEST_COMPLETE_CHECKPOINT_ALIASES = (
    "latest_complete",
    "latest_approved",
)
DEFAULT_CHECKPOINT_ROOT = Path("playground/mowa_ckpt")
DEFAULT_CHECKPOINT_SUBDIR = Path("checkpoints/steps_2")
DEFAULT_FINAL_MODEL_SUBDIR = Path("final_model")


def resolve_mowa_checkpoint_reference(
    repo_root: Path | str,
    reference: Path | str,
    *,
    checkpoint_root_policy: Path | str = DEFAULT_CHECKPOINT_ROOT,
    checkpoint_subdir: Path | str = DEFAULT_CHECKPOINT_SUBDIR,
) -> Path:
    """Resolve explicit paths or latest-complete aliases to repo-relative checkpoint paths."""
    ref = Path(str(reference))
    if str(reference) not in LATEST_COMPLETE_CHECKPOINT_ALIASES:
        return ref
    return _resolve_latest_complete_run(
        Path(repo_root),
        Path(str(checkpoint_root_policy)),
        Path(str(checkpoint_subdir)),
    )


def resolve_mowa_final_model_reference(
    repo_root: Path | str,
    reference: Path | str,
    *,
    checkpoint_root_policy: Path | str = DEFAULT_CHECKPOINT_ROOT,
    final_model_subdir: Path | str = DEFAULT_FINAL_MODEL_SUBDIR,
) -> Path:
    """Resolve explicit paths or latest-complete aliases to repo-relative final-model paths."""
    ref = Path(str(reference))
    if str(reference) not in LATEST_COMPLETE_CHECKPOINT_ALIASES:
        return ref
    latest_checkpoint = _resolve_latest_complete_run(
        Path(repo_root),
        Path(str(checkpoint_root_policy)),
        DEFAULT_CHECKPOINT_SUBDIR,
    )
    return latest_checkpoint.parents[1] / Path(str(final_model_subdir))


def _resolve_latest_complete_run(
    repo_root: Path,
    checkpoint_root: Path,
    checkpoint_subdir: Path,
) -> Path:
    root = repo_root / checkpoint_root
    candidates = []
    if not root.is_dir():
        return checkpoint_root / "__missing_latest_complete__" / checkpoint_subdir
    for run_dir in root.iterdir():
        checkpoint = run_dir / checkpoint_subdir
        completed_steps = _get_completed_steps_if_complete(checkpoint)
        if completed_steps is not None:
            candidates.append((completed_steps, run_dir.name, checkpoint))
    if not candidates:
        return checkpoint_root / "__missing_latest_complete__" / checkpoint_subdir
    _, run_name, _ = sorted(candidates, key=lambda item: (item[0], item[1]))[-1]
    return checkpoint_root / run_name / checkpoint_subdir


def _get_completed_steps_if_complete(checkpoint: Path) -> int | None:
    if not checkpoint.is_dir():
        return None
    required_files = (
        "model.safetensors.index.json",
        "config.full.yaml",
        "dataset_statistics.json",
        "starflow_mapping.json",
        "trainer_state.json",
        "scheduler.pt",
    )
    if not all((checkpoint / name).is_file() for name in required_files):
        return None
    if not any(checkpoint.glob("model-*.safetensors")):
        return None
    if not any(checkpoint.glob("random_states_*.pkl")):
        return None
    if not _has_optimizer_state(checkpoint):
        return None
    trainer_state = _read_json(checkpoint / "trainer_state.json") or {}
    completed_steps = int(trainer_state.get("completed_steps", 0))
    return completed_steps if completed_steps > 0 else None


def _has_optimizer_state(checkpoint: Path) -> bool:
    patterns = (
        "optimizer_rank_*.pt",
        "bf16_zero_pp_rank_*.pt",
        "zero_pp_rank_*.pt",
        "optim_states.pt",
    )
    return any(any(checkpoint.glob(pattern)) for pattern in patterns)


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
