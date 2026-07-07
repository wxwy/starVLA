"""MoWA future-label builder for the TurnOnElectricKettle atomic task."""

from __future__ import annotations

from starVLA.dataloader.mowa.atomic_task_label_builder import register_atomic_task_label_builder
from starVLA.dataloader.mowa.tasks._single_dof import (
    SingleDofTaskBuilder,
    SingleDofTaskSchema,
    select_most_displaced_joint,
)


@register_atomic_task_label_builder
class TurnOnElectricKettleLabelBuilder(SingleDofTaskBuilder):
    """Build future labels for TurnOnElectricKettle using switch joint progress."""

    def __init__(self) -> None:
        super().__init__(
            SingleDofTaskSchema(
                task_name="TurnOnElectricKettle",
                fixture_ref_key="electric_kettle",
                joint_selector=select_most_displaced_joint("switch"),
                handle_site_template=None,
                manipulated_body_template="{fixture_ref}_switch",
                completion_threshold=0.95,
                readiness_distance_threshold=0.05,
                readiness_progress_delta=0.1,
                schema_version="turn_on_electric_kettle_v1",
                subgoal_id="turn_on_electric_kettle",
            )
        )
