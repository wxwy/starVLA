"""MoWA future-label builder for the OpenStandMixerHead atomic task."""

from __future__ import annotations

from starVLA.dataloader.mowa.atomic_task_label_builder import register_atomic_task_label_builder
from starVLA.dataloader.mowa.tasks._single_dof import (
    SingleDofTaskBuilder,
    SingleDofTaskSchema,
    select_most_displaced_joint_matching_all,
)


@register_atomic_task_label_builder
class OpenStandMixerHeadLabelBuilder(SingleDofTaskBuilder):
    """Build future labels for OpenStandMixerHead using head joint progress.

    The stand mixer fixture name varies by episode layout, so we select the
    ``head_joint`` belonging to a ``stand_mixer`` fixture by displacement.
    """

    def __init__(self) -> None:
        super().__init__(
            SingleDofTaskSchema(
                task_name="OpenStandMixerHead",
                fixture_ref_key=None,
                joint_selector=select_most_displaced_joint_matching_all(("stand_mixer", "head_joint")),
                handle_site_template=None,
                manipulated_body_template=None,
                completion_threshold=0.95,
                readiness_distance_threshold=0.05,
                readiness_progress_delta=0.05,
                schema_version="open_stand_mixer_head_v1",
                subgoal_id="open_stand_mixer_head",
            )
        )
