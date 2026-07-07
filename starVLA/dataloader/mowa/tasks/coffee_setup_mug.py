"""MoWA future-label builder for the CoffeeSetupMug atomic task."""

from __future__ import annotations

from starVLA.dataloader.mowa.atomic_task_label_builder import register_atomic_task_label_builder
from starVLA.dataloader.mowa.tasks._object_pose import (
    ObjectPoseTaskBuilder,
    ObjectPoseTaskSchema,
    select_most_displaced_freejoint,
)


@register_atomic_task_label_builder
class CoffeeSetupMugLabelBuilder(ObjectPoseTaskBuilder):
    """Build future labels for CoffeeSetupMug from mug displacement."""

    def __init__(self) -> None:
        super().__init__(
            ObjectPoseTaskSchema(
                task_name="CoffeeSetupMug",
                object_joint_selector=select_most_displaced_freejoint(),
                subgoal_id="coffee_setup_mug",
                completion_threshold=0.95,
                schema_version="coffee_setup_mug_v1",
            )
        )
