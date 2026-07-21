"""Expert action replay test: do dataset actions drive the sim the same way?

Steps the first N expert actions of a training episode through the live sim
and compares the resulting EE/base motion with the recorded states.

判读:
  * sim 运动与数据集轨迹吻合 -> 动作/状态语义在 训练<->仿真 之间一致；
    仿真 0% 应归因于模型闭环质量，而非动作接口。
  * 明显发散 -> 动作语义不匹配（delta vs absolute、坐标系、增益），
    需要回到 convert_hdf5_lerobot / gym_wrapper 链路排查。

Example:
    .robocase/bin/python tools/mowa/e003_expert_replay_sim.py \
        --dataset-dir playground/Datasets/robocasa365/v1.0/target/atomic/OpenDrawer/20250816/lerobot \
        --episode 0 --n-replay 80 --env-name robocasa/OpenDrawer
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import gymnasium as gym  # noqa: E402
import robocasa  # noqa: E402,F401
import robocasa.wrappers.gym_wrapper  # noqa: E402,F401

STATE_KEYS = [
    "state.base_position",
    "state.base_rotation",
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.gripper_qpos",
]

# parquet `action` 列布局（meta/modality.json）：
#   base_motion(0:4), control_mode(4:5), eef_pos(5:8), eef_rot(8:11), gripper_close(11:12)
# parquet `observation.state` 列布局：
#   base_pos(0:3), base_rot(3:7), eef_pos_rel(7:10), eef_rot_rel(10:14), gripper(14:16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--n-replay", type=int, default=80)
    parser.add_argument("--env-name", default="robocasa/OpenDrawer")
    args = parser.parse_args()

    df = pd.read_parquet(
        args.dataset_dir / "data" / "chunk-000" / f"episode_{args.episode:06d}.parquet"
    )
    A = np.stack(df["action"].to_numpy())
    S = np.stack(df["observation.state"].to_numpy())

    env = gym.make(args.env_name, enable_render=False, split="target")
    obs, _ = env.reset()

    def obs_state(o):
        return np.concatenate([np.asarray(o[k]).ravel() for k in STATE_KEYS])

    s0 = obs_state(obs)
    print("=== initial state: sim reset vs dataset frame0 ===")
    names = ["base_pos", "base_rot", "eef_pos_rel", "eef_rot_rel", "grip"]
    slices = [(0, 3), (3, 7), (7, 10), (10, 14), (14, 16)]
    for n, (a, b) in zip(names, slices):
        print(f"  {n:12s} sim={np.round(s0[a:b], 3)}  data={np.round(S[0, a:b], 3)}")

    sim_eef, sim_base = [], []
    success = False
    for t in range(min(args.n_replay, len(df))):
        a = A[t]
        act = {
            "action.end_effector_position": a[5:8],
            "action.end_effector_rotation": a[8:11],
            "action.gripper_close": a[11:12],
            "action.base_motion": a[0:4],
            "action.control_mode": a[4:5],
        }
        obs, _, done, _, info = env.step(act)
        st = obs_state(obs)
        sim_eef.append(st[7:10])
        sim_base.append(st[0:3])
        success |= bool(info.get("success"))
        if done:
            break

    sim_eef = np.array(sim_eef)
    sim_base = np.array(sim_base)
    T = len(sim_eef)
    data_eef = S[1:T + 1, 7:10]
    data_base = S[1:T + 1, 0:3]

    print(f"\n=== replay of {T} expert actions in sim ===")
    print(f"success flag from env: {success}  (场景与数据集 episode 不同，不期望成功)")
    print("\nstep | sim vs data 的 eef 位移（相对各自起点，单位 cm）")
    for t in [1, 2, 4, 8, 16, 32, 48, 64, 80]:
        if t - 1 >= T:
            continue
        ds = (sim_eef[t - 1] - sim_eef[0]) * 100
        dd = (data_eef[t - 1] - data_eef[0]) * 100
        print(f"{t:4d} | sim [{ds[0]:6.1f} {ds[1]:6.1f} {ds[2]:6.1f}]  "
              f"data [{dd[0]:6.1f} {dd[1]:6.1f} {dd[2]:6.1f}]")
    print(f"\nbase motion max|delta|: sim={np.abs(sim_base[-1]-sim_base[0]).max():.4f}  "
          f"data={np.abs(data_base[-1]-data_base[0]).max():.4f}")
    print(f"eef total |delta|: sim={np.abs(sim_eef[-1]-sim_eef[0]).sum():.4f}  "
          f"data={np.abs(data_eef[-1]-data_eef[0]).sum():.4f}")


if __name__ == "__main__":
    main()
