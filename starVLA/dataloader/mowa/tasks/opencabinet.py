"""MoWA future-label builder for the OpenCabinet atomic task."""

from __future__ import annotations

from starVLA.dataloader.mowa.atomic_task_label_builder import register_atomic_task_label_builder
from starVLA.dataloader.mowa.tasks._single_dof import SingleDofTaskBuilder, SingleDofTaskSchema


@register_atomic_task_label_builder
class OpenCabinetLabelBuilder(SingleDofTaskBuilder):
    """Build future labels for OpenCabinet using cabinet door hinge progress."""

    def __init__(self) -> None:
        super().__init__(
            SingleDofTaskSchema(
                task_name="OpenCabinet",
                fixture_ref_key="fxtr",
                joint_name_template="{fixture_ref}_doorhinge",
                handle_site_template="{fixture_ref}_door_handle_default_site",
                manipulated_body_template="{fixture_ref}_door_main",
                completion_threshold=0.95,
                readiness_distance_threshold=0.05,
                readiness_progress_delta=0.05,
                schema_version="opencabinet_open_door_v1",
                subgoal_id="open_cabinet_door",
            )
        )
