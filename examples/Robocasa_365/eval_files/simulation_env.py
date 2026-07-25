"""Run RoboCasa365 (PandaOmron) evaluation against a starVLA websocket policy.

Imports the existing wrappers from ``examples.Robocasa_tabletop`` to avoid
duplication. Differences vs. the GR1 tabletop runner:
  * uses upstream ``robocasa.wrappers.gym_wrapper`` (env id = ``robocasa/<TaskName>``)
  * single-arm 12-d action via ``model2robocasa365_interface.PolicyWarper``
"""

from __future__ import annotations

import dataclasses
import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

# Default to EGL offscreen rendering for RoboCasa eval unless the caller
# explicitly pins a different backend in the environment.
# OSMesa is the historical default but depends on libOSMesa which is not
# always available; EGL works with NVIDIA drivers + libegl1.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# Required for upstream robocasa env registration: ``robocasa/<TaskName>``
import robocasa  # noqa: F401
import robocasa.wrappers.gym_wrapper  # noqa: F401  (registers envs)
import robosuite  # noqa: F401


def _maybe_add_starvla_site_packages_for_client_transport() -> None:
    try:
        import websockets  # noqa: F401
        import msgpack  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    repo_root = Path(__file__).resolve().parents[3]
    py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = repo_root / ".venv" / "lib" / py_version / "site-packages"
    if site_packages.is_dir():
        sys.path.append(str(site_packages))


_maybe_add_starvla_site_packages_for_client_transport()

from examples.Robocasa_365.eval_files.model2robocasa365_interface import PolicyWarper
from examples.Robocasa_tabletop.eval_files.wrappers.multistep_wrapper import MultiStepWrapper
from examples.Robocasa_tabletop.eval_files.wrappers.video_recording_wrapper import (
    VideoRecorder,
    VideoRecordingWrapper,
)


@dataclass
class VideoConfig:
    enabled: bool = True
    video_dir: Optional[str] = None
    steps_per_render: int = 4
    fps: int = 20
    codec: str = "h264"
    input_pix_fmt: str = "rgb24"
    crf: int = 22
    thread_type: str = "FRAME"
    thread_count: int = 2
    record_views: tuple[str, ...] = field(
        default_factory=lambda: ("agentview_left", "eye_in_hand")
    )


@dataclass
class MultiStepConfig:
    video_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    state_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    n_action_steps: int = 8
    max_episode_steps: int = 500


@dataclass
class SimulationConfig:
    env_name: str
    n_episodes: int = 5
    n_envs: int = 1
    split: str = "target"
    seed: Optional[int] = None
    video: VideoConfig = field(default_factory=VideoConfig)
    multistep: MultiStepConfig = field(default_factory=MultiStepConfig)


def _create_single_env(config: SimulationConfig, idx: int) -> gym.Env:
    env = gym.make(config.env_name, enable_render=True, split=config.split)
    if config.video.enabled and config.video.video_dir is not None:
        video_recorder = VideoRecorder.create_h264(
            fps=config.video.fps,
            codec=config.video.codec,
            input_pix_fmt=config.video.input_pix_fmt,
            crf=config.video.crf,
            thread_type=config.video.thread_type,
            thread_count=config.video.thread_count,
        )
        env = VideoRecordingWrapper(
            env,
            video_recorder,
            video_dir=Path(config.video.video_dir),
            steps_per_render=config.video.steps_per_render,
            env_idx=idx,
            record_views=config.video.record_views,
        )
    env = MultiStepWrapper(
        env,
        video_delta_indices=config.multistep.video_delta_indices,
        state_delta_indices=config.multistep.state_delta_indices,
        n_action_steps=config.multistep.n_action_steps,
        max_episode_steps=config.multistep.max_episode_steps,
    )
    return env


def run_simulation(model: PolicyWarper, config: SimulationConfig) -> Tuple[str, List[bool]]:
    print(f"[robocasa365] running {config.n_episodes} eps on {config.env_name}", flush=True)
    env_fns = [partial(_create_single_env, config=config, idx=i) for i in range(config.n_envs)]
    env = (
        gym.vector.SyncVectorEnv(env_fns)
        if config.n_envs == 1
        else gym.vector.AsyncVectorEnv(env_fns, shared_memory=False, context="spawn")
    )

    completed = 0
    episode_successes: List[bool] = []
    current_successes = [False] * config.n_envs
    reset_kwargs = {"seed": config.seed} if config.seed is not None else {}
    obs, _ = env.reset(**reset_kwargs)
    if config.seed is not None:
        print(f"[robocasa365] env reset with base seed={config.seed}", flush=True)
    if model.payload_style == "wanpi":
        task_descriptions = obs["annotation.human.task_description"]
        initial_task = (
            task_descriptions[0]
            if isinstance(task_descriptions, (tuple, list, np.ndarray))
            else task_descriptions
        )
        model.reset(str(initial_task))
        model.observe_wan_frames(obs, initial_only=True)
        zero_actions = {
            key: np.zeros(space.shape, dtype=space.dtype)
            for key, space in env.action_space.spaces.items()
        }
        obs, _, warmup_terminations, warmup_truncations, _ = env.step(zero_actions)
        if np.any(warmup_terminations) or np.any(warmup_truncations):
            raise RuntimeError("RoboCasa episode terminated during Wan VAE causal warmup.")
        print(
            f"[robocasa365] Wan VAE warmed with 1 + {config.multistep.n_action_steps} real frames",
            flush=True,
        )
    t0 = time.time()
    pbar = EpisodeProgressBar(total=config.n_episodes, desc=config.env_name)
    while completed < config.n_episodes:
        out = model.step(obs)
        actions = out["actions"] if "actions" in out else out
        next_obs, rewards, terminations, truncations, env_infos = env.step(actions)
        for i in range(config.n_envs):
            current_successes[i] |= bool(env_infos["success"][i][0])
            if terminations[i] or truncations[i]:
                episode_successes.append(current_successes[i])
                pbar.update(1, success=current_successes[i])
                current_successes[i] = False
                completed += 1
        obs = next_obs

    pbar.close()
    try:
        env.close()
    except Exception as e:  # noqa: BLE001
        print(f"[robocasa365] env.close ignored: {e}")
    print(f"[robocasa365] {completed} eps done in {time.time() - t0:.1f}s")
    return config.env_name, episode_successes


class EpisodeProgressBar:
    """Simple colored progress bar for vectorized episode evaluation."""

    def __init__(self, total: int, desc: str = "", width: int = 30, file=None):
        self.total = max(1, total)
        self.desc = desc
        # Use one block per episode when there are few episodes; cap at ``width``.
        self.width = min(max(10, width), self.total)
        self.file = file or sys.stdout
        self.is_tty = self.file.isatty()
        self.completed = 0
        self.statuses: List[bool] = []  # True for success
        self._last_logged = 0
        self._closed = False

    def update(self, n: int = 1, success: bool = False):
        for _ in range(n):
            if self.completed >= self.total:
                break
            self.completed += 1
            self.statuses.append(bool(success))
        self._render()

    def _block_char(self, start: int, end: int) -> str:
        """Return a single colored block character for episode indices [start, end)."""
        display_completed = min(self.completed, self.total)
        if end <= display_completed:
            block_statuses = self.statuses[start:end]
            if all(block_statuses):
                color, char = ("\033[32m", "█")  # green: all success
            elif not any(block_statuses):
                color, char = ("\033[31m", "█")  # red: all fail
            else:
                color, char = ("\033[33m", "█")  # yellow: mixed
        elif start >= display_completed:
            color, char = ("\033[90m", "░")  # gray: pending
        else:
            color, char = ("\033[33m", "▒")  # yellow: in-progress block
        if self.is_tty:
            return f"{color}{char}\033[0m"
        return char

    def _line(self) -> str:
        display_completed = min(self.completed, self.total)
        successes = sum(self.statuses[:display_completed])
        failures = display_completed - successes
        pct = 100.0 * display_completed / self.total
        bar = "".join(
            self._block_char(
                start=i * self.total // self.width,
                end=(i + 1) * self.total // self.width,
            )
            for i in range(self.width)
        )
        if self.is_tty:
            return (
                f"{self.desc}: |{bar}| "
                f"{display_completed}/{self.total} "
                f"\033[32m✅{successes}\033[0m \033[31m❌{failures}\033[0m "
                f"({pct:.1f}%)"
            )
        return (
            f"{self.desc}: |{bar}| "
            f"{display_completed}/{self.total} "
            f"success={successes} fail={failures} ({pct:.1f}%)"
        )

    def _render(self):
        if self.is_tty:
            self.file.write("\r" + self._line())
            self.file.flush()
            return

        # Non-TTY (e.g. piped to tee): emit a line at most every 5% or 5 episodes.
        step = max(1, self.total // 20) if self.total >= 20 else 1
        if self.completed == self.total or self.completed - self._last_logged >= step:
            print(self._line(), file=self.file, flush=True)
            self._last_logged = self.completed

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.is_tty:
            self.file.write("\n")
            self.file.flush()
        elif self.completed != self._last_logged:
            print(self._line(), file=self.file)


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 5678
    resize_size: tuple = (224, 224)
    env_name: str = "robocasa/OpenDrawer"
    n_episodes: int = 5
    n_envs: int = 1
    split: str = "target"
    max_episode_steps: int = 500
    n_action_steps: int = 8
    wan_history_frames: int = 5
    video_out_path: Optional[str] = "results/robocasa365_eval_test/videos"
    seed: Optional[int] = None
    record_views: tuple[str, ...] = ("agentview_left", "eye_in_hand")
    no_video: bool = False
    payload_style: str = "wanpi"
    pretrained_path: str = (
        "playground/Checkpoints/robocasa365_qwenoft_OpenDrawer_100step/checkpoints/steps_100_pytorch_model.pt"
    )
    unnorm_key: Optional[str] = None


def main(args: Args) -> None:
    # Normalize record_views: tyro passes a comma-separated string as a single
    # tuple element, while argparse splits it. Accept both forms.
    if (
        args.record_views
        and len(args.record_views) == 1
        and isinstance(args.record_views[0], str)
        and "," in args.record_views[0]
    ):
        args.record_views = tuple(x.strip() for x in args.record_views[0].split(","))
    logging.info(json.dumps(dataclasses.asdict(args), indent=2, default=str))
    model = PolicyWarper(
        policy_ckpt_path=args.pretrained_path,
        unnorm_key=args.unnorm_key,
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
        n_action_steps=args.n_action_steps,
        wan_history_frames=args.wan_history_frames,
        payload_style=args.payload_style,
    )
    cfg = SimulationConfig(
        env_name=args.env_name,
        n_episodes=args.n_episodes,
        n_envs=args.n_envs,
        split=args.split,
        seed=args.seed,
        video=VideoConfig(
            enabled=not args.no_video,
            video_dir=args.video_out_path if not args.no_video else None,
            record_views=args.record_views,
        ),
        multistep=MultiStepConfig(
            video_delta_indices=(
                np.array([0])
                if args.payload_style == "groot"
                else np.arange(1 - args.n_action_steps, 1)
            ),
            n_action_steps=args.n_action_steps,
            max_episode_steps=args.max_episode_steps,
        ),
    )
    name, successes = run_simulation(model, cfg)
    sr = float(np.mean(successes)) if successes else 0.0
    print(f"\n=== {name} ===")
    print(f"success rate: {sr:.2f} ({sum(successes)}/{len(successes)})")
    out_dir = Path(args.pretrained_path).with_suffix(".eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{name.replace('/', '_')}.json").open("w") as f:
        json.dump({"env": name, "success_rate": sr, "successes": [bool(s) for s in successes]}, f, indent=2)


def _parse_args_without_tyro() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--args.host", dest="host", default=Args.host)
    parser.add_argument("--args.port", dest="port", type=int, default=Args.port)
    parser.add_argument("--args.env-name", dest="env_name", default=Args.env_name)
    parser.add_argument("--args.split", dest="split", choices=["pretrain", "target"], default=Args.split)
    parser.add_argument("--args.n-episodes", dest="n_episodes", type=int, default=Args.n_episodes)
    parser.add_argument("--args.n-envs", dest="n_envs", type=int, default=Args.n_envs)
    parser.add_argument("--args.max-episode-steps", dest="max_episode_steps", type=int, default=Args.max_episode_steps)
    parser.add_argument("--args.n-action-steps", dest="n_action_steps", type=int, default=Args.n_action_steps)
    parser.add_argument(
        "--args.wan-history-frames",
        dest="wan_history_frames",
        type=int,
        default=Args.wan_history_frames,
    )
    parser.add_argument("--args.video-out-path", dest="video_out_path", default=Args.video_out_path)
    parser.add_argument("--args.seed", dest="seed", type=int, default=Args.seed)
    parser.add_argument(
        "--args.record-views",
        dest="record_views",
        type=lambda s: tuple(x.strip() for x in s.split(",")),
        default=",".join(Args.record_views),
        help="Comma-separated camera views to record, e.g. agentview_left,eye_in_hand",
    )
    parser.add_argument(
        "--args.no-video",
        dest="no_video",
        action="store_true",
        default=Args.no_video,
        help="Disable video recording entirely for faster evaluation.",
    )
    parser.add_argument("--args.pretrained-path", dest="pretrained_path", default=Args.pretrained_path)
    parser.add_argument("--args.unnorm-key", dest="unnorm_key", default=Args.unnorm_key)
    parser.add_argument(
        "--args.payload-style",
        dest="payload_style",
        choices=["wanpi", "starflow"],
        default=Args.payload_style,
        help="wanpi: E-003 WanPI 双视角5帧历史; starflow: E-001 baseline 3视角单帧",
    )
    parsed = parser.parse_args()
    return Args(**vars(parsed))


if __name__ == "__main__":
    try:
        import tyro

        tyro.cli(main)
    except ModuleNotFoundError as exc:
        if exc.name != "tyro":
            raise
        main(_parse_args_without_tyro())
