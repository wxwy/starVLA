"""LIBERO benchmark — data config, embodiment tags, and mixtures."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionSinCosTransform,
    StateActionToTensor,
    StateActionTransform,
)
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------
class Libero4in1DataConfig:
    embodiment_tag = EmbodimentTag.FRANKA
    video_keys = [
        "video.primary_image",
        "video.wrist_image",
    ]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.pad",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(8))
    state_indices = [0]

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys), # igore state modality for now since some datasets don't have state and we want to be able to use them, can add back later if needed
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.x": "min_max",
                    "action.y": "min_max",
                    "action.z": "min_max",
                    "action.roll": "min_max",
                    "action.pitch": "min_max",
                    "action.yaw": "min_max",
                },
            ),
        ])


class Libero4in1WanPIDataConfig:
    """LIBERO 4-in-1 configuration for the MoWA E-003 WanPI path."""

    embodiment_tag = EmbodimentTag.FRANKA
    video_keys = [
        "video.primary_image",
        "video.wrist_image",
    ]
    # LIBERO observation.state is eef_pos(3) + axis_angle(3) + gripper_qpos(2).
    # WanPI uses the same sin/cos state injection as the RoboCasa E-003 path.
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.pad",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    action_key_dims = {
        "action.x": 1,
        "action.y": 1,
        "action.z": 1,
        "action.roll": 1,
        "action.pitch": 1,
        "action.yaw": 1,
        "action.gripper": 1,
    }
    state_key_dims = {
        "state.x": 1,
        "state.y": 1,
        "state.z": 1,
        "state.roll": 1,
        "state.pitch": 1,
        "state.yaw": 1,
        "state.pad": 1,
        "state.gripper": 1,
    }
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(8))
    state_indices = [0]

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionSinCosTransform(apply_to=self.state_keys),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.x": "min_max",
                    "action.y": "min_max",
                    "action.z": "min_max",
                    "action.roll": "min_max",
                    "action.pitch": "min_max",
                    "action.yaw": "min_max",
                    "action.gripper": "binary",
                },
            ),
        ])


ROBOT_TYPE_CONFIG_MAP = {
    "libero_franka": Libero4in1DataConfig(),
    "libero_franka_wanpi": Libero4in1WanPIDataConfig(),
}


# ---------------------------------------------------------------------------
# Embodiment Tags
# ---------------------------------------------------------------------------
ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    # Per Proposal A, embodiment_tag now lives as a classvar on each DataConfig.
    # The registry derives ROBOT_TYPE_TO_EMBODIMENT_TAG automatically. Kept as
    # an empty dict for backward compat (it is honored as legacy override).
}


# ---------------------------------------------------------------------------
# Mixtures
# ---------------------------------------------------------------------------
DATASET_NAMED_MIXTURES = {
    "libero_all": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_goal": [
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_4in1_wanpi": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka_wanpi"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka_wanpi"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka_wanpi"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka_wanpi"),
    ],
    "multi_robot": [
        ("LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
}
