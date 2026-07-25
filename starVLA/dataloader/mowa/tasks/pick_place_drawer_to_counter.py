"""MoWA future-label builder for PickPlaceDrawerToCounter."""

from __future__ import annotations

from starVLA.dataloader.mowa.atomic_task_label_builder import register_atomic_task_label_builder
from starVLA.dataloader.mowa.tasks._object_pose import (
    ObjectPoseTaskBuilder,
    ObjectPoseTaskSchema,
    select_most_displaced_freejoint,
)


@register_atomic_task_label_builder
class PickPlaceDrawerToCounterLabelBuilder(ObjectPoseTaskBuilder):
    """Build future labels for PickPlaceDrawerToCounter from object displacement."""

    def __init__(self) -> None:
        super().__init__(
            ObjectPoseTaskSchema(
                task_name="PickPlaceDrawerToCounter",
                object_joint_selector=select_most_displaced_freejoint(),
                subgoal_id="pick_place_drawer_to_counter",
                completion_threshold=0.95,
                schema_version="pick_place_drawer_to_counter_v1",
            )
        )
