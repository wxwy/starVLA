# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import uuid
from pathlib import Path
from typing import Optional, Sequence

import av
import gymnasium as gym
import numpy as np


def get_accumulate_timestamp_idxs(
    timestamps: list[float],
    start_time: float,
    dt: float,
    eps: float = 1e-5,
    next_global_idx: int | None = 0,
    allow_negative: bool = False,
) -> tuple[list[int], list[int], int]:
    """
    For each dt window, choose the first timestamp in the window.
    Assumes timestamps sorted. One timestamp might be chosen multiple times due to dropped frames.
    next_global_idx should start at 0 normally, and then use the returned next_global_idx.
    However, when overwiting previous values are desired, set last_global_idx to None.

    Returns:
    local_idxs: which index in the given timestamps array to chose from
    global_idxs: the global index of each chosen timestamp
    next_global_idx: used for next call.
    """
    local_idxs = list()
    global_idxs = list()
    for local_idx, ts in enumerate(timestamps):
        # add eps * dt to timestamps so that when ts == start_time + k * dt
        # is always recorded as kth element (avoiding floating point errors)
        global_idx = np.floor((ts - start_time) / dt + eps)
        if (not allow_negative) and (global_idx < 0):
            continue
        if next_global_idx is None:
            next_global_idx = global_idx

        n_repeats = max(0, global_idx - next_global_idx + 1)
        for i in range(n_repeats):
            local_idxs.append(local_idx)
            global_idxs.append(next_global_idx + i)
        next_global_idx += n_repeats
    return local_idxs, global_idxs, next_global_idx


class VideoRecorder:
    def __init__(
        self,
        fps,
        codec,
        input_pix_fmt,
        # options for codec
        **kwargs,
    ):
        """
        input_pix_fmt: rgb24, bgr24 see https://github.com/PyAV-Org/PyAV/blob/bc4eedd5fc474e0f25b22102b2771fe5a42bb1c7/av/video/frame.pyx#L352
        """

        self.fps = fps
        self.codec = codec
        self.input_pix_fmt = input_pix_fmt
        self.kwargs = kwargs
        # runtime set
        self._reset_state()

    def _reset_state(self):
        self.container = None
        self.stream = None
        self.shape = None
        self.dtype = None
        self.start_time = None
        self.next_global_idx = 0

    @classmethod
    def create_h264(
        cls,
        fps,
        codec="h264",
        input_pix_fmt="rgb24",
        output_pix_fmt="yuv420p",
        crf=18,
        profile="high",
        **kwargs,
    ):
        obj = cls(
            fps=fps,
            codec=codec,
            input_pix_fmt=input_pix_fmt,
            pix_fmt=output_pix_fmt,
            options={"crf": str(crf), "profile:v": "high"},
            **kwargs,
        )
        return obj

    def __del__(self):
        self.stop()

    def is_ready(self):
        return self.stream is not None

    def start(self, file_path, start_time=None):
        if self.is_ready():
            # if still recording, stop first and start anew.
            self.stop()

        self.container = av.open(file_path, mode="w")
        self.stream = self.container.add_stream(self.codec, rate=self.fps)
        codec_context = self.stream.codec_context
        for k, v in self.kwargs.items():
            setattr(codec_context, k, v)
        self.start_time = start_time

    def write_frame(self, img: np.ndarray, frame_time=None):
        if not self.is_ready():
            raise RuntimeError("Must run start() before writing!")

        n_repeats = 1
        if self.start_time is not None:
            local_idxs, global_idxs, self.next_global_idx = get_accumulate_timestamp_idxs(
                # only one timestamp
                timestamps=[frame_time],
                start_time=self.start_time,
                dt=1 / self.fps,
                next_global_idx=self.next_global_idx,
            )
            # number of appearance means repeats
            n_repeats = len(local_idxs)

        if self.shape is None:
            self.shape = img.shape
            self.dtype = img.dtype
            h, w, c = img.shape
            self.stream.width = w
            self.stream.height = h
        assert img.shape == self.shape
        assert img.dtype == self.dtype

        frame = av.VideoFrame.from_ndarray(img, format=self.input_pix_fmt)
        for i in range(n_repeats):
            for packet in self.stream.encode(frame):
                self.container.mux(packet)

    def stop(self):
        if not self.is_ready():
            return

        # Flush stream
        for packet in self.stream.encode():
            self.container.mux(packet)

        # Close the file
        self.container.close()

        # reset runtime parameters
        self._reset_state()


class VideoRecordingWrapper(gym.Wrapper):
    """Record episodes to video, optionally concatenating multiple camera views."""

    # Friendly aliases -> raw RoboSuite camera names used by RoboCasa.
    VIEW_NAME_MAP = {
        "agentview_left": "robot0_agentview_left",
        "main_left": "robot0_agentview_left",
        "agentview_right": "robot0_agentview_right",
        "main_right": "robot0_agentview_right",
        "eye_in_hand": "robot0_eye_in_hand",
        "wrist": "robot0_eye_in_hand",
    }

    def __init__(
        self,
        env,
        video_recorder: VideoRecorder,
        mode="rgb_array",
        video_dir: Path | None = None,
        steps_per_render=1,
        task_description: Optional[str] = None,
        env_idx: int = 0,
        record_views: Sequence[str] = ("agentview_left", "eye_in_hand"),
        **kwargs,
    ):
        """
        When file_path is None, don't record.

        Args:
            task_description: Optional human-readable task description used in the
                video filename. If not provided, the wrapper will try to extract it
                from the observation dict returned by ``reset()`` (e.g.
                ``annotation.human.task_description`` for RoboCasa environments).
            env_idx: Environment index, included in the filename when no task
                description is available.
            record_views: Ordered list of camera views to concatenate horizontally.
                Supported aliases: ``agentview_left``/``main_left``,
                ``agentview_right``/``main_right``, ``eye_in_hand``/``wrist``.
                Raw RoboSuite camera names are also accepted. Default is
                ``("agentview_left", "eye_in_hand")``.
        """
        super().__init__(env)

        if video_dir is not None:
            video_dir.mkdir(parents=True, exist_ok=True)

        self.mode = mode
        self.render_kwargs = kwargs
        self.steps_per_render = steps_per_render
        self.video_dir = video_dir
        self.video_recorder = video_recorder
        self.file_path = None

        self.step_count = 0

        self.is_success = False

        self.task_description = task_description
        self.env_idx = env_idx
        self.episode_idx = 0
        self.record_views = tuple(record_views)

    @staticmethod
    def _sanitize_filename(s: str) -> str:
        """Replace spaces/special characters with underscores for safe filenames."""
        # Keep word chars and CJK chars, collapse everything else to a single underscore.
        sanitized = re.sub(r"[^\w一-鿿]+", "_", s.strip())
        return sanitized.strip("_")

    def _build_filename(self) -> str:
        if self.task_description:
            task_part = self._sanitize_filename(self.task_description)
        else:
            task_part = f"env{self.env_idx}"
        filename = f"{task_part}_episode{self.episode_idx:03d}.mp4"
        return filename

    def _find_base_env(self):
        """Traverse wrappers to reach the base RoboCasa/RoboSuite env."""
        env = self.env
        while isinstance(env, gym.Wrapper):
            env = env.env
        return env

    @staticmethod
    def _match_shape(img: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Crop or pad ``img`` to ``target_shape`` (H, W, C)."""
        if img.shape == target_shape:
            return img
        h, w = target_shape[:2]
        ih, iw = img.shape[:2]
        # Crop if larger.
        y0 = max(0, (ih - h) // 2)
        x0 = max(0, (iw - w) // 2)
        img = img[y0 : y0 + min(h, ih), x0 : x0 + min(w, iw)]
        # Pad if smaller.
        ph = max(0, h - img.shape[0])
        pw = max(0, w - img.shape[1])
        if ph or pw:
            img = np.pad(
                img,
                ((0, ph), (0, pw), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        return img

    def _render_view(self, view_name: str, raw_obs: Optional[dict] = None) -> np.ndarray | None:
        """Render a single camera view, matching the orientation of ``env.render()``.

        Prefer frames that are already present in the environment's last observation
        dict (cheap), otherwise fall back to an explicit render call.
        """
        camera_name = self.VIEW_NAME_MAP.get(view_name, view_name)

        # 1) Try to reuse the already-computed frame from the env observation.
        video_key = f"video.{camera_name}"
        if isinstance(raw_obs, dict) and video_key in raw_obs:
            frame = raw_obs[video_key]
            if isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.shape[-1] == 3:
                return np.ascontiguousarray(frame).astype(np.uint8)

        # 2) Fall back to explicit rendering.
        base = self._find_base_env()
        if base is None or not hasattr(base, "camera_names"):
            return None

        if camera_name not in base.camera_names:
            return None

        # The default env.render() already returns the first camera, vertically flipped.
        render_obs_key = getattr(base, "render_obs_key", None)
        if render_obs_key == f"{camera_name}_image":
            return self.env.render()

        width = getattr(base, "camera_widths", None)
        height = getattr(base, "camera_heights", None)
        if isinstance(width, (list, tuple, np.ndarray)):
            idx = base.camera_names.index(camera_name)
            width = width[idx]
            height = height[idx]
        elif width is None:
            # Fallback to the render() frame size if dimensions are unknown.
            ref = self.env.render()
            height, width = ref.shape[:2]

        if not hasattr(base, "env"):
            return None
        try:
            frame = base.env.sim.render(int(height), int(width), camera_name=camera_name)
        except Exception:
            return None

        if frame is None:
            return None

        # RoboCasa wrapper flips images vertically before exposing them; do the same.
        return np.ascontiguousarray(frame[::-1, :, :]).astype(np.uint8)

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        # Gymnasium reset returns (obs, info); older APIs may return just obs.
        obs = result[0] if isinstance(result, tuple) else result

        self.frames = list()
        self.step_count = 1
        self.video_recorder.stop()

        if self.video_dir is not None and self.file_path is not None:
            # rename the file to indicate success or failure
            original_filestem = self.file_path.stem
            new_filestem = f"{original_filestem}_success{int(self.is_success)}"
            new_file_path = self.video_dir / f"{new_filestem}.mp4"
            os.rename(self.file_path, new_file_path)

        # Auto-extract task description from observation dict (e.g. RoboCasa).
        # Update on every reset so that episodes with varying language instructions
        # (e.g. "Open the left drawer" vs "Open the right drawer") get correct filenames.
        if isinstance(obs, dict):
            for key in (
                "annotation.human.task_description",
                "task_description",
                "language_instruction",
            ):
                if key in obs:
                    self.task_description = obs[key]
                    break

        self.is_success = False
        if self.video_dir is not None:
            self.file_path = self.video_dir / self._build_filename()
            self.episode_idx += 1
        return result

    def step(self, action):
        result = super().step(action)
        self.step_count += 1
        # Track episode-level success: OR over all steps so the filename suffix
        # matches the success recorded in eval_report.json (which uses any-step success).
        self.is_success |= bool(result[-1]["success"])
        if self.file_path is not None and ((self.step_count % self.steps_per_render) == 0):
            if not self.video_recorder.is_ready():
                self.video_recorder.start(self.file_path)

            # Reuse the raw observation dict returned by the wrapped env.
            # MultiStepWrapper stacks historical frames, but the inner env still returns
            # the current raw dict as result[0]; it already contains all camera views.
            raw_obs = result[0] if isinstance(result, tuple) else result
            if not isinstance(raw_obs, dict):
                raw_obs = None

            frames = []
            for view_name in self.record_views:
                frame = self._render_view(view_name, raw_obs=raw_obs)
                if frame is not None:
                    frames.append(frame)

            if not frames:
                # Fallback to the default render if none of the configured views exist.
                frames = [self.env.render()]

            # Normalize heights before horizontal concatenation.
            target_shape = frames[0].shape
            frames = [self._match_shape(f, target_shape) for f in frames]
            frame = np.concatenate(frames, axis=1)
            assert frame.dtype == np.uint8
            self.video_recorder.write_frame(frame)
        return result

    def render(self, mode="rgb_array", **kwargs):
        if self.video_recorder.is_ready():
            self.video_recorder.stop()
        return self.file_path
