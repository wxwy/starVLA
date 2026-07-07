"""MoWA future-label builder for the TurnOnMicrowave atomic task."""

from __future__ import annotations

from starVLA.dataloader.mowa.atomic_task_label_builder import register_atomic_task_label_builder
from starVLA.dataloader.mowa.tasks._reward_based import RewardBasedTaskBuilder, RewardBasedTaskSchema


@register_atomic_task_label_builder
class TurnOnMicrowaveLabelBuilder(RewardBasedTaskBuilder):
    """Build future labels for TurnOnMicrowave from the binary reward signal.

    The microwave start button is not exposed as a moving joint in the episode
    XML, so we use the reward signal as a coarse progress proxy.
    """

    def __init__(self) -> None:
        super().__init__(
            RewardBasedTaskSchema(
                task_name="TurnOnMicrowave",
                subgoal_id="turn_on_microwave",
                completion_threshold=0.95,
                schema_version="turn_on_microwave_v1",
            )
        )
