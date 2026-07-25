"""MoWA future-label builder for PickPlaceCounterToStove."""

from __future__ import annotations

from starVLA.dataloader.mowa.atomic_task_label_builder import register_atomic_task_label_builder
from starVLA.dataloader.mowa.tasks._object_pose import (
    ObjectPoseTaskBuilder,
    ObjectPoseTaskSchema,
    select_most_displaced_freejoint,
)


@register_atomic_task_label_builder
class PickPlaceCounterToStoveLabelBuilder(ObjectPoseTaskBuilder):
    """Build future labels for PickPlaceCounterToStove from object displacement."""

    def __init__(self) -> None:
        super().__init__(
            ObjectPoseTaskSchema(
                task_name="PickPlaceCounterToStove",
                object_joint_selector=select_most_displaced_freejoint(),
                subgoal_id="pick_place_counter_to_stove",
                completion_threshold=0.95,
                schema_version="pick_place_counter_to_stove_v1",
            )
        )
