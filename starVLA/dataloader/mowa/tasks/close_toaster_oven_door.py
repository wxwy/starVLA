"""MoWA future-label builder for the CloseToasterOvenDoor atomic task."""

from __future__ import annotations

from starVLA.dataloader.mowa.atomic_task_label_builder import register_atomic_task_label_builder
from starVLA.dataloader.mowa.tasks._single_dof import SingleDofTaskBuilder, SingleDofTaskSchema


@register_atomic_task_label_builder
class CloseToasterOvenDoorLabelBuilder(SingleDofTaskBuilder):
    """Build future labels for CloseToasterOvenDoor using toaster-oven door progress."""

    def __init__(self) -> None:
        super().__init__(
            SingleDofTaskSchema(
                task_name="CloseToasterOvenDoor",
                fixture_ref_key="toaster_oven",
                joint_name_template="{fixture_ref}_door_joint",
                handle_site_template="{fixture_ref}_door_handle_default_site",
                manipulated_body_template="{fixture_ref}_door",
                invert_progress=True,
                completion_threshold=0.95,
                readiness_distance_threshold=0.05,
                readiness_progress_delta=0.05,
                schema_version="close_toaster_oven_door_v1",
                subgoal_id="close_toaster_oven_door",
            )
        )
