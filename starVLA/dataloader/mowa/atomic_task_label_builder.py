"""Task-agnostic base class and registry for MoWA atomic-task future labels.

Each atomic task implements an ``AtomicTaskLabelBuilder`` subclass and registers
it.  The production dataloader and offline precompute tools then discover the
builder by task name (parsed from the dataset path) rather than hardcoding task
logic.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from starVLA.mowa_constants import MOWA_FUTURE_FULL_HEADS


class AtomicTaskLabelBuilder(ABC):
    """Abstract builder for per-episode future labels on one atomic task.

    Subclasses must implement ``task_name`` and ``build_cache_for_episode``.
    They may override ``sidecar_dir_name`` and ``required_extras`` if the
    defaults are not appropriate.
    """

    @property
    @abstractmethod
    def task_name(self) -> str:
        """Canonical task name matching the registry and dataset paths."""

    @abstractmethod
    def build_cache_for_episode(
        self,
        parquet_path: Path,
        states_path: Path,
        model_path: Path | None = None,
        ep_meta_path: Path | None = None,
        *,
        failure_risk_horizon: int = 10,
        subgoal_horizon: int = 20,
        readiness_horizon: int = 5,
        readiness_progress_delta: float = 0.1,
        readiness_distance_threshold: float = 0.05,
        enable_kinematics: bool = True,
        repo_root: Path | None = None,
    ) -> dict[str, Any]:
        """Compute tensor-ready labels/masks for one episode.

        Must return a dict containing at least the standard sidecar columns:
        ``frame_index``, ``failure_risk``, ``failure_risk_mask``,
        ``subgoal_feasibility``, ``subgoal_feasibility_mask``,
        ``manipulation_readiness``, ``manipulation_readiness_mask``.

        It may also include task-specific debug columns and metadata keys such
        as ``row_count``, ``schema_version``, ``readiness_schema_version``,
        ``readiness_predicate``, ``kinematics_available``.
        """

    def sidecar_dir_name(self) -> str:
        """Directory name under ``<dataset>/mowa_future_labels/``."""
        return self.task_name

    def required_extras(self) -> tuple[str, ...]:
        """Extra filenames required per episode (relative to ``extras/episode_X/``)."""
        return ("states.npz", "ep_meta.json", "model.xml.gz")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_ATOMIC_TASK_BUILDERS: dict[str, type[AtomicTaskLabelBuilder]] = {}


def register_atomic_task_label_builder(cls: type[AtomicTaskLabelBuilder]) -> type[AtomicTaskLabelBuilder]:
    """Decorator that registers an ``AtomicTaskLabelBuilder`` subclass."""
    name = cls().task_name
    if name in _ATOMIC_TASK_BUILDERS:
        raise ValueError(f"Atomic task label builder already registered for {name}")
    _ATOMIC_TASK_BUILDERS[name] = cls
    return cls


def get_registered_atomic_task_names() -> tuple[str, ...]:
    """Return all registered task names in insertion order."""
    return tuple(_ATOMIC_TASK_BUILDERS.keys())


def get_builder_for_task(task_name: str) -> AtomicTaskLabelBuilder:
    """Instantiate the registered builder for ``task_name``."""
    cls = _ATOMIC_TASK_BUILDERS.get(task_name)
    if cls is None:
        raise KeyError(f"No AtomicTaskLabelBuilder registered for task {task_name}")
    return cls()


def is_task_registered(task_name: str) -> bool:
    """Return True if a builder is registered for ``task_name``."""
    return task_name in _ATOMIC_TASK_BUILDERS


# ---------------------------------------------------------------------------
# Dataset-path helpers
# ---------------------------------------------------------------------------

_DATASET_PATH_TASK_RE = re.compile(r"/target/atomic/([^/]+)/\d{8}/lerobot")


def get_task_name_from_dataset_path(dataset_path: Path | str) -> str | None:
    """Infer the atomic task name from a LeRobot dataset path.

    Example path:
        .../v1.0/target/atomic/OpenDrawer/20250816/lerobot
    Returns:
        "OpenDrawer"
    """
    match = _DATASET_PATH_TASK_RE.search(str(dataset_path))
    if match:
        return match.group(1)
    return None


def get_builder_for_dataset_path(dataset_path: Path | str) -> AtomicTaskLabelBuilder | None:
    """Look up the builder for a dataset path, or None if not registered."""
    task_name = get_task_name_from_dataset_path(dataset_path)
    if task_name is None or not is_task_registered(task_name):
        return None
    return get_builder_for_task(task_name)


# ---------------------------------------------------------------------------
# Shared label helpers
# ---------------------------------------------------------------------------

def failure_risk_label_from_window(
    rewards: list[float],
    dones: list[bool],
    anchor: int,
    horizon: int,
) -> float | None:
    """Shared H-step failure-risk proxy.

    Returns 0.0 if a ``done`` with ``reward > 0`` occurs within the window,
    1.0 if a ``done`` with ``reward <= 0`` occurs first, and None if no
    termination is observed (censored sample).
    """
    for reward, done in zip(rewards[anchor + 1 : anchor + 1 + horizon], dones[anchor + 1 : anchor + 1 + horizon]):
        if not done:
            continue
        return 0.0 if reward > 0 else 1.0
    return None


def compute_subgoal_feasibility(
    progress: np.ndarray,
    completion_threshold: float,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute binary subgoal feasibility labels and masks.

    For each timestep, the label is 1.0 if ``progress`` reaches
    ``completion_threshold`` within the next ``horizon`` steps and the subgoal
    is not already completed.  The mask is True (meaning the label should be
    masked) when the subgoal is already completed at the anchor step.

    Returns (values, masks) where masks are True for anchors that should be
    masked.  This matches the convention in ``opendrawer_label_cache.py``.
    """
    row_count = int(progress.shape[0])
    values = np.zeros(row_count, dtype=np.float64)
    masks = np.zeros(row_count, dtype=bool)
    for timestep in range(row_count):
        current_progress = float(progress[timestep])
        completed_now = current_progress >= completion_threshold
        masks[timestep] = completed_now
        if not completed_now:
            future = progress[timestep + 1 : timestep + 1 + horizon]
            if future.size > 0 and np.any(future >= completion_threshold):
                values[timestep] = 1.0
    return values, masks


def compute_manipulation_readiness_from_progress(
    progress: np.ndarray,
    progress_delta: float,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute readiness labels from future progress gain (progress-imminence proxy).

    Returns (values, masks).  An anchor is masked if the subgoal is already
    completed or if the future window is empty.
    """
    row_count = int(progress.shape[0])
    values = np.zeros(row_count, dtype=np.float64)
    masks = np.ones(row_count, dtype=bool)
    for timestep in range(row_count):
        current_progress = float(progress[timestep])
        future = progress[timestep + 1 : timestep + 1 + horizon]
        if future.size == 0:
            continue
        masks[timestep] = current_progress >= 1.0
        if masks[timestep]:
            continue
        if np.any(future - current_progress >= progress_delta):
            values[timestep] = 1.0
    return values, masks


def make_default_masks(unmasked_heads: tuple[str, ...] | None = None) -> dict[str, bool]:
    """Return a mask dict where only ``unmasked_heads`` are True.

    Defaults to the constructible heads defined in ``mowa_constants.py``.
    """
    from starVLA.mowa_constants import MOWA_FUTURE_CONSTRUCTIBLE_HEADS

    if unmasked_heads is None:
        unmasked_heads = MOWA_FUTURE_CONSTRUCTIBLE_HEADS
    return {head: head in unmasked_heads for head in MOWA_FUTURE_FULL_HEADS}

