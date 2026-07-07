"""MoWA future-label builder for the SlideDishwasherRack atomic task."""

from __future__ import annotations

from starVLA.dataloader.mowa.atomic_task_label_builder import register_atomic_task_label_builder
from starVLA.dataloader.mowa.tasks._single_dof import (
    SingleDofTaskBuilder,
    SingleDofTaskSchema,
    select_most_displaced_joint,
)


@register_atomic_task_label_builder
class SlideDishwasherRackLabelBuilder(SingleDofTaskBuilder):
    """Build future labels for SlideDishwasherRack using rack slide progress.

    The target rack (rack0/rack1) can vary by episode layout, so we select the
    rack joint with the largest qpos displacement.
    """

    def __init__(self) -> None:
        super().__init__(
            SingleDofTaskSchema(
                task_name="SlideDishwasherRack",
                fixture_ref_key="dishwasher",
                joint_selector=select_most_displaced_joint("rack"),
                handle_site_template=None,
                manipulated_body_template="{fixture_ref}_rack0",
                completion_threshold=0.95,
                readiness_distance_threshold=0.05,
                readiness_progress_delta=0.1,
                schema_version="slide_dishwasher_rack_v1",
                subgoal_id="slide_dishwasher_rack",
            )
        )
