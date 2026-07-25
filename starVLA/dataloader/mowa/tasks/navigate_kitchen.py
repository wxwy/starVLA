"""MoWA future-label builder for the NavigateKitchen atomic task."""

from __future__ import annotations

from starVLA.dataloader.mowa.atomic_task_label_builder import register_atomic_task_label_builder
from starVLA.dataloader.mowa.tasks._reward_based import RewardBasedTaskBuilder, RewardBasedTaskSchema


@register_atomic_task_label_builder
class NavigateKitchenLabelBuilder(RewardBasedTaskBuilder):
    """Build future labels for NavigateKitchen from the binary reward signal.

    Without MuJoCo forward-kinematics we cannot compute Euclidean distance from
    the robot base to the target fixture, so we fall back to the reward signal
    as a coarse progress proxy.
    """

    def __init__(self) -> None:
        super().__init__(
            RewardBasedTaskSchema(
                task_name="NavigateKitchen",
                subgoal_id="navigate_kitchen",
                completion_threshold=0.95,
                schema_version="navigate_kitchen_v1",
            )
        )
