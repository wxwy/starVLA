"""Shared MoWA constants that must not import torch or dataset code."""

MOWA_FUTURE_CONSTRUCTIBLE_HEADS = ("task_progress", "action_outcome_class")

MOWA_FUTURE_FULL_HEADS = (
    "task_progress",
    "manipulation_readiness",
    "failure_risk",
    "next_best_view_score",
    "subgoal_feasibility",
    "object_visibility_future",
    "action_outcome_class",
)

MOWA_FUTURE_MASKED_HEADS = tuple(
    head for head in MOWA_FUTURE_FULL_HEADS if head not in MOWA_FUTURE_CONSTRUCTIBLE_HEADS
)

MOWA_FUTURE_HEAD_OUTPUT_DIMS = {
    "task_progress": 1,
    "manipulation_readiness": 1,
    "failure_risk": 1,
    "next_best_view_score": 1,
    "subgoal_feasibility": 1,
    "object_visibility_future": 1,
    "action_outcome_class": 2,
}

MOWA_STARFLOW_CONDITION_PROBE_FEATURE_SOURCE = "starflow_condition_probe"
MOWA_FUTURE_FEATURE_HEADS_SOURCE = "mowa_future_feature_heads"
MOWA_FUTURE_FEATURE_SOURCE_ALIASES = (MOWA_FUTURE_FEATURE_HEADS_SOURCE,)

MOWA_ACTION_OUTCOME_CLASS_MAPPING_STATUS = "confirmed_for_e001_initial_target"
MOWA_ACTION_OUTCOME_CLASS_MAPPING_VERSION = "reward_done_vector_v1"
MOWA_ACTION_OUTCOME_CLASS_MAPPING_NOTE = (
    "action_outcome_class uses [next_reward, next_done_flag] in this order."
)

# Atomic tasks with local target/human LeRobot data in the RoboCasa365 registry.
# These names match the directory names under
# ``v1.0/target/atomic/<Task>/<date>/lerobot`` and the keys used by
# ``AtomicTaskLabelBuilder`` subclasses.
MOWA_ATOMIC_TASK_NAMES = (
    "CloseBlenderLid",
    "CloseFridge",
    "CloseToasterOvenDoor",
    "CoffeeSetupMug",
    "NavigateKitchen",
    "OpenCabinet",
    "OpenDrawer",
    "OpenStandMixerHead",
    "PickPlaceCounterToCabinet",
    "PickPlaceCounterToStove",
    "PickPlaceDrawerToCounter",
    "PickPlaceSinkToCounter",
    "PickPlaceToasterToCounter",
    "SlideDishwasherRack",
    "TurnOffStove",
    "TurnOnElectricKettle",
    "TurnOnMicrowave",
    "TurnOnSinkFaucet",
)
