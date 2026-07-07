"""MoWA OpenDrawer future-label pre-computation cache.

.. deprecated::
    This module is kept for backward compatibility.  New code should use the
    generic framework in ``atomic_task_label_builder.py``,
    ``label_cache.py``, and ``tasks/opendrawer.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# Import tasks to ensure the OpenDrawer builder is registered.
import starVLA.dataloader.mowa.tasks  # noqa: F401
from starVLA.dataloader.mowa.atomic_task_label_builder import (
    get_builder_for_task,
)
from starVLA.dataloader.mowa.label_cache import (
    build_label_cache_for_episode as _build_label_cache_for_episode,
    label_cache_available as _label_cache_available,
    load_label_cache_for_episode as _load_label_cache_for_episode,
    write_label_cache as _write_label_cache,
)


OPENDRAWER_LABEL_CACHE_DIR = "mowa_future_labels/OpenDrawer"

OPENDRAWER_LABEL_CACHE_COLUMNS = (
    "frame_index",
    "drawer_progress",
    "eef_to_handle_distance",
    "gripper_handle_contact",
    "gripper_finger_contact",
    "failure_risk",
    "failure_risk_mask",
    "subgoal_feasibility",
    "subgoal_feasibility_mask",
    "manipulation_readiness",
    "manipulation_readiness_mask",
)


def build_opendrawer_label_cache_for_episode(
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
    """Compute tensor-ready labels/masks for one OpenDrawer episode.

    Deprecated: delegates to the generic OpenDrawer task builder.
    """
    builder = get_builder_for_task("OpenDrawer")
    return _build_label_cache_for_episode(
        builder=builder,
        parquet_path=parquet_path,
        states_path=states_path,
        model_path=model_path,
        ep_meta_path=ep_meta_path,
        failure_risk_horizon=failure_risk_horizon,
        subgoal_horizon=subgoal_horizon,
        readiness_horizon=readiness_horizon,
        readiness_progress_delta=readiness_progress_delta,
        readiness_distance_threshold=readiness_distance_threshold,
        enable_kinematics=enable_kinematics,
        repo_root=repo_root,
    )


def write_opendrawer_label_cache(
    dataset_path: Path | str,
    output_dir: Path | str | None = None,
    *,
    failure_risk_horizon: int = 10,
    subgoal_horizon: int = 20,
    readiness_horizon: int = 5,
    readiness_progress_delta: float = 0.1,
    readiness_distance_threshold: float = 0.05,
    enable_kinematics: bool = True,
    max_episodes: int | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Pre-compute OpenDrawer future labels for all episodes and write sidecars.

    Deprecated: delegates to the generic sidecar writer.
    """
    builder = get_builder_for_task("OpenDrawer")
    if output_dir is None:
        output_dir = Path(dataset_path) / OPENDRAWER_LABEL_CACHE_DIR
    return _write_label_cache(
        builder=builder,
        dataset_path=dataset_path,
        output_dir=output_dir,
        failure_risk_horizon=failure_risk_horizon,
        subgoal_horizon=subgoal_horizon,
        readiness_horizon=readiness_horizon,
        readiness_progress_delta=readiness_progress_delta,
        readiness_distance_threshold=readiness_distance_threshold,
        enable_kinematics=enable_kinematics,
        max_episodes=max_episodes,
        repo_root=repo_root,
    )


def load_opendrawer_label_cache_for_episode(
    dataset_path: Path | str,
    episode_index: int,
) -> dict[str, Any] | None:
    """Load a pre-computed OpenDrawer label sidecar for one episode.

    Deprecated: delegates to the generic sidecar loader.
    """
    builder = get_builder_for_task("OpenDrawer")
    return _load_label_cache_for_episode(
        builder=builder,
        dataset_path=dataset_path,
        episode_index=episode_index,
    )


def opendrawer_label_cache_available(dataset_path: Path | str) -> bool:
    """Return True if at least one OpenDrawer episode sidecar exists.

    Deprecated: delegates to the generic sidecar availability check.
    """
    builder = get_builder_for_task("OpenDrawer")
    return _label_cache_available(builder=builder, dataset_path=dataset_path)
