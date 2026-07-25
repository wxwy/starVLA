# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""LIBERO env-side adapter (thin client).

After the server-side refactor (see `deployment/model_server/policy_wrapper.py`),
the websocket *server* now returns already-unnormalized actions and ships
model-invariant fields (`action_chunk_size`, `available_unnorm_keys`) at
handshake. This client therefore no longer needs to:
  - load `dataset_statistics.json`
  - know `future_action_window_size`
  - perform un-normalization

It only handles env-specific adaptation: image history bookkeeping, action
ensembling, gripper sticky logic, and chunk-cache scheduling.
"""

from collections import deque
import time
from typing import Optional, Sequence
import uuid

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.SimplerEnv.eval_files.adaptive_ensemble import AdaptiveEnsembler


class ModelClient:
    def __init__(
        self,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "franka",
        horizon: int = 0,
        replan_interval: Optional[int] = None,
        action_ensemble: bool = True,
        action_ensemble_horizon: Optional[int] = 3,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha: float = 0.1,
        host: str = "0.0.0.0",
        port: int = 10095,
        image_size: Sequence[int] = (224, 224),
        payload_style: str = "standard",
        wan_history_frames: int = 5,
    ) -> None:
        # Connect & receive handshake metadata (action_chunk_size, etc.)
        self.client = WebsocketClientPolicy(host, port)
        meta = self.client.get_server_metadata()
        self.action_chunk_size = int(meta["action_chunk_size"])
        self._server_metadata = meta
        self.replan_interval = (
            self.action_chunk_size if replan_interval is None else max(1, min(int(replan_interval), self.action_chunk_size))
        )

        self.image_size: tuple = tuple(image_size)
        if payload_style not in ("standard", "wanpi"):
            raise ValueError(f"payload_style must be standard/wanpi, got {payload_style!r}")
        if payload_style == "wanpi" and (wan_history_frames < 5 or (wan_history_frames - 1) % 4):
            raise ValueError("wan_history_frames must be 1 + 4k and at least 5.")
        self.payload_style = payload_style
        self.wan_history_frames = int(wan_history_frames)
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key
        print(
            f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key}, "
            f"action_chunk_size: {self.action_chunk_size}, replan_interval: {self.replan_interval}, "
            f"payload_style: {payload_style}, server_meta: {meta} ***"
        )

        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.horizon = horizon
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon

        # Gripper sticky state (kept for parity with the previous client; not
        # currently consumed by LIBERO but other policy_setup paths use it).
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        self._wan_stream_id = f"libero-{uuid.uuid4().hex}"
        self._wan_pending_views = [[], []]
        self._wan_stream_reset = True
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(
                self.action_ensemble_horizon, self.adaptive_ensemble_alpha
            )
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        # Cached unnormalized chunk; refreshed every `action_chunk_size` steps.
        self.raw_actions: Optional[np.ndarray] = None
        self._last_chunk_timings: Optional[dict] = None
        self._current_chunk_start_step: int = 0

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        self._wan_pending_views = [[], []]
        self._wan_stream_reset = True
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.raw_actions = None
        self._last_chunk_timings = None
        self._current_chunk_start_step = 0

    def observe_wan_images(self, images: Sequence[np.ndarray]) -> None:
        """缓存自上次策略请求以来的真实双视角新帧。"""
        if self.payload_style != "wanpi":
            return
        if not isinstance(images, (list, tuple)) or len(images) != 2:
            raise ValueError("LIBERO WanPI requires [main_image, wrist_image].")
        target_hw = self.image_size
        for view_index, image in enumerate(images):
            array = np.asarray(image)
            if target_hw and array.shape[:2] != target_hw:
                array = np.asarray(
                    Image.fromarray(array).resize(
                        (target_hw[1], target_hw[0]), Image.BILINEAR
                    )
                )
            self._wan_pending_views[view_index].append(array)

    def step(self, example: dict, step: int = 0, **kwargs) -> dict:
        """One env step.

        Args:
            example: dict with keys ``image`` (list of np.uint8 HWC arrays) and ``lang`` (str).
            step: env step counter; used for chunk caching.

        Returns:
            ``{"raw_action": {"world_vector": ..., "rotation_delta": ..., "open_gripper": ...}}``
        """
        task_description = example.get("lang", None)
        if task_description != self.task_description:
            self.reset(task_description)

        resize_start = time.perf_counter()
        # Resize images to self.image_size if needed.
        if self.image_size and example.get("image"):
            resized = []
            target_hw = self.image_size  # (H, W)
            for img in example["image"]:
                arr = np.asarray(img)
                if arr.shape[:2] != target_hw:
                    arr = np.asarray(
                        Image.fromarray(arr).resize(
                            (target_hw[1], target_hw[0]), Image.BILINEAR
                        )
                    )
                resized.append(arr)
            example = {**example, "image": resized}
        resize_elapsed = time.perf_counter() - resize_start

        if self.payload_style == "wanpi":
            images = example.get("image")
            self.observe_wan_images(images)
            raw_state = np.asarray(example.get("state"), dtype=np.float32)
            if raw_state.ndim == 1:
                raw_state = raw_state[None, :]
            if raw_state.ndim != 2 or raw_state.shape[-1] != 8:
                raise ValueError(f"LIBERO WanPI expects raw state shape (1, 8), got {raw_state.shape}")
            resized_example = {
                "mowa_multi_view_images": [list(frames) for frames in self._wan_pending_views],
                "mowa_stream_id": self._wan_stream_id,
                "mowa_stream_reset": self._wan_stream_reset,
                "lang": example.get("lang", ""),
                "state": self._sin_cos_state(raw_state),
            }
        else:
            resized_example = example
        # Refresh chunk if needed. `replan_interval=1` means re-run the model
        # every env step and only consume the latest chunk's first action.
        cache_refresh = self.raw_actions is None or (step - self._current_chunk_start_step) >= self.replan_interval
        if cache_refresh:
            vla_input = {
                "examples": [resized_example],
                "unnorm_key": self.unnorm_key,
                "do_sample": False,
                "use_ddim": self.use_ddim,
                "num_ddim_steps": self.num_ddim_steps,
            }
            response = self.client.predict_action(vla_input)
            if self.payload_style == "wanpi":
                self._wan_pending_views = [[], []]
                self._wan_stream_reset = False
            try:
                actions_batch = response["data"]["actions"]  # (B, T, D), unnormalized server-side
            except KeyError:
                raise KeyError(
                    f"Key 'actions' not found in response data: keys={list(response.get('data', {}).keys())}, "
                    f"full response={response}"
                )
            self.raw_actions = np.asarray(actions_batch)[0]  # (T, D)
            self._current_chunk_start_step = step
            server_timings = response.get("data", {}).get("timings", {})
            self._last_chunk_timings = {
                "cache_refresh": True,
                "chunk_step": step,
                "replan_interval": self.replan_interval,
                "resize_sec": resize_elapsed,
                "client_pack_and_queue_sec": response.get("_client_timings", {}).get("pack_and_queue_sec"),
                "client_server_roundtrip_sec": response.get("_client_timings", {}).get("server_roundtrip_sec"),
                "client_unpack_sec": response.get("_client_timings", {}).get("unpack_sec"),
                "client_total_call_sec": response.get("_client_timings", {}).get("total_client_call_sec"),
                "server_total_sec": server_timings.get("server_total_sec"),
                "server_framework_sec": server_timings.get("framework_sec"),
                "server_unnorm_sec": server_timings.get("unnorm_sec"),
                "framework_prepare_inputs_sec": server_timings.get("framework_prepare_inputs_sec"),
                "framework_build_qwen_inputs_sec": server_timings.get("framework_build_qwen_inputs_sec"),
                "framework_qwen_forward_sec": server_timings.get("framework_qwen_forward_sec"),
                "framework_gather_action_tokens_sec": server_timings.get("framework_gather_action_tokens_sec"),
                "framework_action_head_sec": server_timings.get("framework_action_head_sec"),
                "framework_to_numpy_sec": server_timings.get("framework_to_numpy_sec"),
                "framework_total_sec": server_timings.get("framework_total_sec"),
            }
        action_index = step - self._current_chunk_start_step
        raw_actions = self.raw_actions[action_index][None]
        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # 1 = open; 0 = close
        }
        return {
            "raw_action": raw_action,
            "timings": {
                "cache_refresh": cache_refresh,
                "chunk_step": self._current_chunk_start_step,
                "action_index": action_index,
                "replan_interval": self.replan_interval,
                "resize_sec": resize_elapsed,
                "chunk_request": self._last_chunk_timings if cache_refresh else None,
            },
        }

    @staticmethod
    def _sin_cos_state(state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32)
        return np.concatenate([np.sin(state), np.cos(state)], axis=-1)

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]
        img_strip = np.concatenate(np.array(images[::3]), axis=1)
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)
