"""MoWA future-label builder for the CloseBlenderLid atomic task."""

from __future__ import annotations

import numpy as np

from starVLA.dataloader.mowa.atomic_task_label_builder import register_atomic_task_label_builder
from starVLA.dataloader.mowa.tasks._single_dof import (
    SingleDofTaskBuilder,
    SingleDofTaskSchema,
    select_most_displaced_joint_matching_all,
)


def _blender_lid_progress_normalizer(
    raw_qpos: np.ndarray, joint_range: tuple[float, ...]
) -> np.ndarray:
    """Normalize a blender lid joint without a fixed range.

    The lid joint is continuous (no range in XML).  We use the observed qpos
    range within the episode: high qpos means open (progress 0), low qpos
    means closed (progress 1).
    """
    lo = float(np.min(raw_qpos))
    hi = float(np.max(raw_qpos))
    denom = hi - lo
    if denom <= 0:
        return raw_qpos
    return (hi - raw_qpos) / denom


@register_atomic_task_label_builder
class CloseBlenderLidLabelBuilder(SingleDofTaskBuilder):
    """Build future labels for CloseBlenderLid using blender lid joint progress."""

    def __init__(self) -> None:
        super().__init__(
            SingleDofTaskSchema(
                task_name="CloseBlenderLid",
                fixture_ref_key=None,
                joint_selector=select_most_displaced_joint_matching_all(("blender", "auxiliary")),
                handle_site_template=None,
                manipulated_body_template=None,
                progress_normalizer=_blender_lid_progress_normalizer,
                completion_threshold=0.95,
                readiness_distance_threshold=0.05,
                readiness_progress_delta=0.05,
                schema_version="close_blender_lid_v2",
                subgoal_id="close_blender_lid",
            )
        )
