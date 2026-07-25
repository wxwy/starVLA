"""EXT-B1 Phase 0A: official GR00T N1.5 native rollout smoke (single task).

Loads the official checkpoint with the official Gr00tPolicy (data_config
panda_omron), starts the official RobotInferenceServer in a thread, and runs
a thin client over an explicit task list (the official run_eval.py only
accepts whole TASK_SET_REGISTRY sets).

Usage (in the ext_b1_groot venv):
    python tools/mowa/mowa_external_groot_n15_native_smoke.py \
        --model-path /disk/rl/models/robocasa365_baselines/e000_b1_gr00t_n1_5/gr00t_n1-5/multitask_learning/checkpoint-120000 \
        --tasks OpenDrawer --split target --n-episodes 2 --max-episode-steps 750 \
        --output-dir playground/mowa_eval_results/ext_b1_groot_n15/phase0_native

NOTE: per the design's statistics section, 0/N here is NOT a failure verdict;
this phase only proves checkpoint + env + inference path work end to end.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "playground" / "Code" / "Isaac-GR00T"))

from gr00t.eval.robot import RobotInferenceServer  # noqa: E402
from gr00t.eval.simulation import (  # noqa: E402
    MultiStepConfig,
    SimulationConfig,
    SimulationInferenceClient,
    VideoConfig,
)
from gr00t.experiment.data_config import DATA_CONFIG_MAP  # noqa: E402
from gr00t.model.policy import Gr00tPolicy  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--data-config", default="panda_omron")
    parser.add_argument("--tasks", nargs="+", default=["OpenDrawer"])
    parser.add_argument("--split", choices=["pretrain", "target"], default="target")
    parser.add_argument("--n-episodes", type=int, default=2)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--n-action-steps", type=int, default=16)
    parser.add_argument("--max-episode-steps", type=int, default=750)
    parser.add_argument("--port", type=int, default=5557)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    data_config = DATA_CONFIG_MAP[args.data_config]
    print(f"[ext-b1] loading Gr00tPolicy from {args.model_path}", flush=True)
    policy = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag=args.embodiment_tag,
        denoising_steps=4,
    )
    print(f"[ext-b1] policy loaded in {time.time()-t0:.1f}s", flush=True)

    server = RobotInferenceServer(policy, port=args.port)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()
    time.sleep(2)

    client = SimulationInferenceClient(host="localhost", port=args.port)
    report = {"model_path": args.model_path, "split": args.split, "tasks": {}}
    for env_name in args.tasks:
        this_video_dir = None if args.no_video else str(args.output_dir / "videos" / env_name)
        config = SimulationConfig(
            env_name=f"robocasa/{env_name}",
            split=args.split,
            n_episodes=args.n_episodes,
            n_envs=args.n_envs,
            video=VideoConfig(video_dir=this_video_dir) if this_video_dir else None,
            multistep=MultiStepConfig(
                n_action_steps=args.n_action_steps,
                max_episode_steps=args.max_episode_steps,
            ),
        )
        print(f"[ext-b1] running robocasa/{env_name} ({args.n_episodes} eps) ...", flush=True)
        t1 = time.time()
        name, successes = client.run_simulation(config)
        sr = float(np.mean(successes)) if successes else 0.0
        print(f"[ext-b1] {name}: SR={sr:.2f} ({sum(successes)}/{len(successes)}) "
              f"in {time.time()-t1:.1f}s", flush=True)
        report["tasks"][env_name] = {
            "success_rate": sr,
            "successes": [bool(s) for s in successes],
            "elapsed_sec": time.time() - t1,
        }

    out = args.output_dir / "phase0a_native_smoke.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"[ext-b1] report -> {out}", flush=True)


if __name__ == "__main__":
    main()
