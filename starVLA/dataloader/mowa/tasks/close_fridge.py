"""MoWA future-label builder for the CloseFridge atomic task."""

from __future__ import annotations

from starVLA.dataloader.mowa.atomic_task_label_builder import register_atomic_task_label_builder
from starVLA.dataloader.mowa.tasks._single_dof import (
    SingleDofTaskBuilder,
    SingleDofTaskSchema,
    select_most_displaced_joint_matching_all,
)


@register_atomic_task_label_builder
class CloseFridgeLabelBuilder(SingleDofTaskBuilder):
    """Build future labels for CloseFridge using the most displaced fridge door.

    Episodes may feature a single fridge door or a pair of French doors.  We
    track the door with the largest qpos displacement and invert the progress
    so that ``closed = 1.0``.
    """

    def __init__(self) -> None:
        super().__init__(
            SingleDofTaskSchema(
                task_name="CloseFridge",
                fixture_ref_key=None,
                joint_selector=select_most_displaced_joint_matching_all(("fridge", "door_joint")),
                handle_site_template=None,
                manipulated_body_template=None,
                invert_progress=True,
                completion_threshold=0.95,
                readiness_distance_threshold=0.05,
                readiness_progress_delta=0.1,
                schema_version="close_fridge_v1",
                subgoal_id="close_fridge",
            )
        )
