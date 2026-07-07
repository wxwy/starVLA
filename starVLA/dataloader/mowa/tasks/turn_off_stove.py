"""MoWA future-label builder for the TurnOffStove atomic task."""

from __future__ import annotations

from starVLA.dataloader.mowa.atomic_task_label_builder import register_atomic_task_label_builder
from starVLA.dataloader.mowa.tasks._single_dof import (
    SingleDofTaskBuilder,
    SingleDofTaskSchema,
    select_most_displaced_joint,
)


@register_atomic_task_label_builder
class TurnOffStoveLabelBuilder(SingleDofTaskBuilder):
    """Build future labels for TurnOffStove using stove knob joint progress.

    The active burner knob varies by episode, so we select the knob joint with
    the largest qpos displacement.
    """

    def __init__(self) -> None:
        super().__init__(
            SingleDofTaskSchema(
                task_name="TurnOffStove",
                fixture_ref_key=None,
                joint_selector=select_most_displaced_joint("stove_main_group_knob_"),
                handle_site_template=None,
                manipulated_body_template=None,
                completion_threshold=0.95,
                readiness_distance_threshold=0.05,
                readiness_progress_delta=0.1,
                schema_version="turn_off_stove_v1",
                subgoal_id="turn_off_stove",
            )
        )
