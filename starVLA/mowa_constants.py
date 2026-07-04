"""Shared MoWA constants that must not import torch or dataset code."""

MOWA_P0_CONSTRUCTIBLE_HEADS = ("task_progress", "action_outcome_class")

MOWA_P0_FULL_HEADS = (
    "task_progress",
    "manipulation_readiness",
    "failure_risk",
    "next_best_view_score",
    "subgoal_feasibility",
    "object_visibility_future",
    "action_outcome_class",
)

MOWA_P0_MASKED_HEADS = tuple(
    head for head in MOWA_P0_FULL_HEADS if head not in MOWA_P0_CONSTRUCTIBLE_HEADS
)

MOWA_P0_HEAD_OUTPUT_DIMS = {
    "task_progress": 1,
    "manipulation_readiness": 1,
    "failure_risk": 1,
    "next_best_view_score": 1,
    "subgoal_feasibility": 1,
    "object_visibility_future": 1,
    "action_outcome_class": 2,
}

# Semantic aliases for runtime code. The P0 names remain as compatibility
# aliases for existing configs, reports and experiment-stage documents.
MOWA_FUTURE_CONSTRUCTIBLE_HEADS = MOWA_P0_CONSTRUCTIBLE_HEADS
MOWA_FUTURE_FULL_HEADS = MOWA_P0_FULL_HEADS
MOWA_FUTURE_MASKED_HEADS = MOWA_P0_MASKED_HEADS
MOWA_FUTURE_HEAD_OUTPUT_DIMS = MOWA_P0_HEAD_OUTPUT_DIMS
