"""Policy-side bridge for RoboCasa365 (PandaOmron, single-arm) evaluation.

Modeled after ``examples/Robocasa_tabletop/eval_files/model2robocasa_interface.py``
but adapted to the single-arm 12-d action / 16-d state layout produced by
``robocasa.wrappers.gym_wrapper.PandaOmronKeyConverter``.
"""

from typing import Dict, Optional
import uuid

import cv2 as cv
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.Robocasa_tabletop.eval_files.adaptive_ensemble import AdaptiveEnsembler


# Order MUST match the LeRobot dataset ``observation.state`` produced by
# ``robocasa/scripts/dataset_scripts/convert_hdf5_lerobot.py``:
#   base_position(3) + base_rotation(4) + eef_pos_rel(3) + eef_rot_rel(4) + gripper_qpos(2) = 16
STATE_KEY_ORDER = [
    "state.base_position",
    "state.base_rotation",
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.gripper_qpos",
]

# Action splits in the trained 12-d output (see PandaOmronRoboCasa365DataConfig)
ACTION_SLICES = {
    "action.end_effector_position": (0, 3),
    "action.end_effector_rotation": (3, 6),
    "action.gripper_close": (6, 7),
    "action.base_motion": (7, 11),
    "action.control_mode": (11, 12),
}


class PolicyWarper:
    """Single-arm PandaOmron policy wrapper that talks to the websocket server."""

    def __init__(
        self,
        policy_ckpt_path: str,
        unnorm_key: Optional[str] = None,
        host: str = "0.0.0.0",
        port: int = 10095,
        image_size=(224, 224),
        n_action_steps: int = 8,
        action_ensemble: bool = False,
        action_ensemble_horizon: int = 3,
        adaptive_ensemble_alpha: float = 0.1,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        wan_history_frames: int = 5,
        payload_style: str = "wanpi",
    ) -> None:
        self.client = WebsocketClientPolicy(host, port)
        self.unnorm_key = unnorm_key
        self.image_size = tuple(image_size)
        self.n_action_steps = n_action_steps
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        if payload_style not in ("wanpi", "starflow", "groot"):
            raise ValueError(f"payload_style must be wanpi/starflow/groot, got {payload_style!r}")
        self.payload_style = payload_style
        if payload_style == "groot":
            # EXT-B1: GR00T 单帧观测，无 wan 历史帧约束；原始 256² 与原始 16 维 state 直接透传。
            self.wan_history_frames = 1
        else:
            if wan_history_frames < 5 or (wan_history_frames - 1) % 4:
                raise ValueError("wan_history_frames must be 1 + 4k and at least 5.")
            self.wan_history_frames = wan_history_frames
            if payload_style == "wanpi" and n_action_steps % 4:
                raise ValueError("WanPI n_action_steps must be divisible by 4 for causal VAE streaming.")

        self.task_description = None
        self._wan_stream_namespace = f"robocasa-{uuid.uuid4().hex}"
        self._wan_pending_views = None
        self._wan_stream_reset = None
        self.action_ensemble = action_ensemble
        self.action_ensembler = (
            AdaptiveEnsembler(action_ensemble_horizon, adaptive_ensemble_alpha)
            if action_ensemble
            else None
        )

        server_meta = self.client.get_server_metadata()
        print(f"*** unnorm_key: {unnorm_key}, server_meta: {server_meta} ***")
        self._validate_server_metadata(server_meta)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def reset(self, task_description) -> None:
        self.task_description = task_description
        self._wan_pending_views = None
        self._wan_stream_reset = None
        if self.action_ensemble:
            self.action_ensembler.reset()

    def _ensure_wan_stream_batch(self, batch_size: int) -> None:
        if self._wan_pending_views is None or len(self._wan_pending_views) != batch_size:
            self._wan_pending_views = [[[], []] for _ in range(batch_size)]
            self._wan_stream_reset = [True] * batch_size

    def observe_wan_frames(self, observations: Dict, *, initial_only: bool = False) -> None:
        """缓存每个环境自上次策略请求以来的真实双视角新帧。"""
        if self.payload_style != "wanpi":
            return
        main_view = observations["video.robot0_agentview_left"]
        wrist_view = observations["video.robot0_eye_in_hand"]
        if len(main_view) != len(wrist_view):
            raise ValueError("RoboCasa main/wrist batch sizes are not synchronized.")
        self._ensure_wan_stream_batch(len(main_view))
        for batch_index, (main_frames, wrist_frames) in enumerate(zip(main_view, wrist_view, strict=True)):
            if len(main_frames) != len(wrist_frames):
                raise ValueError("RoboCasa main/wrist frame counts are not synchronized.")
            if initial_only:
                main_frames = main_frames[-1:]
                wrist_frames = wrist_frames[-1:]
            self._wan_pending_views[batch_index][0].extend(list(main_frames))
            self._wan_pending_views[batch_index][1].extend(list(wrist_frames))

    def step(self, observations: Dict, **_) -> Dict:
        # 0) data-flow contract check (mirrors StarFlow/E003 validation style)
        self._validate_observations(observations)

        # 1) instruction
        task_descs = observations["annotation.human.task_description"]  # tuple of B strs
        if isinstance(task_descs, (tuple, list)):
            instructions = list(task_descs)
        else:
            instructions = [task_descs]
        if instructions[0] != self.task_description:
            self.reset(instructions[0])

        # 2) image + state — 按模型训练时的模态构造 payload。
        if self.payload_style == "groot":
            # EXT-B1 (GR00T N1.5)：3 视角原始 uint8 256² 单帧（client 不做任何
            # resize/crop，官方 processor 独占预处理）；state_raw 为原始 16 维
            # （不做 sin/cos，官方 transform 内部做 rotation_6d + min_max）。
            view_keys = (
                "video.robot0_agentview_left",
                "video.robot0_agentview_right",
                "video.robot0_eye_in_hand",
            )
            latest_frames = []  # per view: list over batch of raw (256,256,3) uint8
            for view_key in view_keys:
                stacked = observations[view_key]  # (B, T, H, W, 3)
                latest_frames.append([sample[-1] for sample in stacked])
            n_batch = len(latest_frames[0])
            state_parts = [observations[k] for k in STATE_KEY_ORDER]  # each (B, 1, d)
            input_state = np.concatenate(state_parts, axis=-1)  # (B, 1, 16)
            examples = [
                {
                    "image": [latest_frames[v][b] for v in range(3)],
                    "lang": instructions[b] if b < len(instructions) else instructions[0],
                    "state_raw": input_state[b],
                }
                for b in range(n_batch)
            ]
        elif self.payload_style == "starflow":
            # StarFlowVLA (E-001 baseline) 训练用 3 视角单帧 (left/right/wrist)，
            # 在线取各视角最新帧并缩放到训练分辨率。
            view_keys = (
                "video.robot0_agentview_left",
                "video.robot0_agentview_right",
                "video.robot0_eye_in_hand",
            )
            latest_frames = []  # per view: list over batch of resized (H, W, 3) frames
            for view_key in view_keys:
                stacked = observations[view_key]  # (B, T, H, W, 3)
                latest_frames.append(
                    [self._resize_image(sample[-1]) for sample in stacked]
                )
            n_batch = len(latest_frames[0])
            state_parts = [observations[k] for k in STATE_KEY_ORDER]  # each (B, 1, d)
            input_state = self._sin_cos_state(np.concatenate(state_parts, axis=-1))
            examples = [
                {
                    "image": [latest_frames[v][b] for v in range(3)],
                    "lang": instructions[b] if b < len(instructions) else instructions[0],
                    "state": input_state[b],
                }
                for b in range(n_batch)
            ]
        else:
            # main/wrist 各自维持 episode 级因果 VAE stream；这里只发送新增帧。
            self.observe_wan_frames(observations)
            assert self._wan_pending_views is not None and self._wan_stream_reset is not None

            # 3) state — concatenate parts in the same order as in training
            state_parts = [observations[k] for k in STATE_KEY_ORDER]  # each (B, 1, d)
            input_state = np.concatenate(state_parts, axis=-1)  # (B, 1, 16)
            input_state = self._sin_cos_state(input_state)

            examples = []
            for b in range(len(self._wan_pending_views)):
                examples.append(
                    {
                        "mowa_multi_view_images": [
                            list(self._wan_pending_views[b][0]),
                            list(self._wan_pending_views[b][1]),
                        ],
                        "mowa_stream_id": f"{self._wan_stream_namespace}:{b}",
                        "mowa_stream_reset": self._wan_stream_reset[b],
                        "lang": instructions[b] if b < len(instructions) else instructions[0],
                        "state": input_state[b],
                    }
                )

        vla_input = {
            "examples": examples,
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }
        vla_input["unnorm_key"] = self.unnorm_key
        response = self.client.predict_action(vla_input)
        if self.payload_style == "wanpi":
            self._wan_pending_views = [[[], []] for _ in self._wan_pending_views]
            self._wan_stream_reset = [False] * len(self._wan_stream_reset)
        # server already un-normalized via training-time transform
        raw_actions = np.array(response["data"]["actions"])  # (B, chunk, D)
        self._validate_action_response(raw_actions)

        if self.action_ensemble:
            ensembled = []
            for b in range(raw_actions.shape[0]):
                ensembled.append(self.action_ensembler.ensemble_action(raw_actions[b])[None])
            raw_actions = np.stack(ensembled, axis=0)

        # Slice into the dict structure consumed by RoboCasaGymEnv.step().
        out = {}
        for key, (s, e) in ACTION_SLICES.items():
            out[key] = raw_actions[:, : self.n_action_steps, s:e]
        return {"actions": out}

    # ------------------------------------------------------------------
    # Data-flow validation (referenced from StarFlow/E003 contracts)
    # ------------------------------------------------------------------
    def _validate_server_metadata(self, server_meta: Dict) -> None:
        """Fail fast if the server's expected embodiment keys do not match ours."""
        server_state_keys = server_meta.get("state_keys")
        server_action_keys = server_meta.get("action_keys")
        action_chunk_size = server_meta.get("action_chunk_size")

        if server_state_keys is not None and list(server_state_keys) != list(STATE_KEY_ORDER):
            raise ValueError(
                f"Server state_keys mismatch: server={server_state_keys}, "
                f"client expects={STATE_KEY_ORDER}. Check that the checkpoint "
                "was trained with the PandaOmronRoboCasa365 embodiment."
            )

        if server_action_keys is not None and list(server_action_keys) != list(ACTION_SLICES.keys()):
            raise ValueError(
                f"Server action_keys mismatch: server={server_action_keys}, "
                f"client expects={list(ACTION_SLICES.keys())}."
            )

        if action_chunk_size is not None and self.n_action_steps > int(action_chunk_size):
            raise ValueError(
                f"n_action_steps={self.n_action_steps} cannot exceed server's "
                f"action_chunk_size={action_chunk_size}."
            )

    def _validate_observations(self, observations: Dict) -> None:
        """Ensure all modalities required by the model are present and well-formed."""
        # 1) instruction
        if "annotation.human.task_description" not in observations:
            raise ValueError("Missing observation key: annotation.human.task_description")
        task_descs = observations["annotation.human.task_description"]
        if isinstance(task_descs, (tuple, list)):
            instructions = list(task_descs)
        else:
            instructions = [task_descs]
        for i, text in enumerate(instructions):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Invalid task description at index {i}: {text!r}")

        if self.payload_style == "groot":
            # EXT-B1：只需确认 3 视角与 state 键存在且形状可解释；具体内容（分辨率、
            # 归一化）由官方 processor 负责，client 不做二次校验。
            for view_key in (
                "video.robot0_agentview_left",
                "video.robot0_agentview_right",
                "video.robot0_eye_in_hand",
            ):
                if view_key not in observations:
                    raise ValueError(f"Missing observation key: {view_key}")
            for key in STATE_KEY_ORDER:
                if key not in observations:
                    raise ValueError(f"Missing observation key: {key}")
            return

        # 2) main/wrist video
        for view_key in ("video.robot0_agentview_left", "video.robot0_eye_in_hand"):
            if view_key not in observations:
                raise ValueError(f"Missing observation key: {view_key}")
        main_view = observations["video.robot0_agentview_left"]
        wrist_view = observations["video.robot0_eye_in_hand"]
        if not isinstance(main_view, np.ndarray) or not isinstance(wrist_view, np.ndarray):
            raise ValueError(
                f"Video observations must be np.ndarray, got main={type(main_view)}, "
                f"wrist={type(wrist_view)}"
            )
        if main_view.shape != wrist_view.shape:
            raise ValueError(
                f"Main/wrist view shape mismatch: main={main_view.shape}, wrist={wrist_view.shape}"
            )
        if main_view.ndim != 5 or main_view.shape[-1] != 3:
            raise ValueError(
                f"Expected video shape (B, T, H, W, 3), got {main_view.shape}"
            )
        if main_view.shape[1] != self.n_action_steps:
            raise ValueError(
                f"Expected video time dim={self.n_action_steps}, got {main_view.shape[1]}"
            )
        if not np.isfinite(main_view).all() or not np.isfinite(wrist_view).all():
            raise ValueError("Video observations contain NaN or Inf.")

        # 3) state
        for key in STATE_KEY_ORDER:
            if key not in observations:
                raise ValueError(f"Missing observation key: {key}")
        state_parts = [observations[k] for k in STATE_KEY_ORDER]
        batch_sizes = {part.shape[0] for part in state_parts}
        if len(batch_sizes) != 1:
            raise ValueError(f"Inconsistent state batch sizes across keys: {batch_sizes}")
        input_state = np.concatenate(state_parts, axis=-1)
        if input_state.ndim != 3 or input_state.shape[1] != 1:
            raise ValueError(
                f"Expected state shape (B, 1, D), got {input_state.shape}"
            )
        if not np.isfinite(input_state).all():
            raise ValueError("State observations contain NaN or Inf.")

        # 4) cross-modality batch size consistency
        if main_view.shape[0] != input_state.shape[0]:
            raise ValueError(
                f"Batch size mismatch: video={main_view.shape[0]}, state={input_state.shape[0]}"
            )

    def _validate_action_response(self, raw_actions: np.ndarray) -> None:
        """Ensure the server returned a well-formed action chunk."""
        if not isinstance(raw_actions, np.ndarray):
            raise ValueError(f"Expected np.ndarray actions, got {type(raw_actions)}")
        if raw_actions.ndim != 3:
            raise ValueError(f"Expected actions shape (B, chunk, D), got {raw_actions.shape}")
        if raw_actions.shape[1] < self.n_action_steps:
            raise ValueError(
                f"Server returned action chunk length={raw_actions.shape[1]}, "
                f"but n_action_steps={self.n_action_steps}"
            )
        if raw_actions.shape[2] != sum(e - s for s, e in ACTION_SLICES.values()):
            raise ValueError(
                f"Server action dim={raw_actions.shape[2]} does not match expected "
                f"action dim={sum(e - s for s, e in ACTION_SLICES.values())}"
            )
        if not np.isfinite(raw_actions).all():
            raise ValueError("Server actions contain NaN or Inf.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        return cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)

    @staticmethod
    def _sin_cos_state(state: np.ndarray) -> np.ndarray:
        """Match training-time StateActionSinCosTransform on the state."""
        return np.concatenate([np.sin(state), np.cos(state)], axis=-1)
