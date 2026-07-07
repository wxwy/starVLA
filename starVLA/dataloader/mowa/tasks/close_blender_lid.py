"""MoWA future-label builder for the CloseBlenderLid atomic task."""

from __future__ import annotations

from starVLA.dataloader.mowa.atomic_task_label_builder import register_atomic_task_label_builder
from starVLA.dataloader.mowa.tasks._single_dof import (
    SingleDofTaskBuilder,
    SingleDofTaskSchema,
    select_most_displaced_joint_matching_all,
)


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
                completion_threshold=0.95,
                readiness_distance_threshold=0.05,
                readiness_progress_delta=0.1,
                schema_version="close_blender_lid_v1",
                subgoal_id="close_blender_lid",
            )
        )
