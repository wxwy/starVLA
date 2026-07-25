"""MoWA future-label builder for PickPlaceSinkToCounter."""

from __future__ import annotations

from starVLA.dataloader.mowa.atomic_task_label_builder import register_atomic_task_label_builder
from starVLA.dataloader.mowa.tasks._object_pose import (
    ObjectPoseTaskBuilder,
    ObjectPoseTaskSchema,
    select_most_displaced_freejoint,
)


@register_atomic_task_label_builder
class PickPlaceSinkToCounterLabelBuilder(ObjectPoseTaskBuilder):
    """Build future labels for PickPlaceSinkToCounter from object displacement."""

    def __init__(self) -> None:
        super().__init__(
            ObjectPoseTaskSchema(
                task_name="PickPlaceSinkToCounter",
                object_joint_selector=select_most_displaced_freejoint(),
                subgoal_id="pick_place_sink_to_counter",
                completion_threshold=0.95,
                schema_version="pick_place_sink_to_counter_v1",
            )
        )
