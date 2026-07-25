"""旧版 (2026-04-26, commit 342d3da) server 协议的 WM4A 离线动作对照。

与 tools/mowa/wm4a_offline_action_check.py 相同的数据与探针，但使用旧协议：
response["data"]["normalized_actions"] + 客户端 min_max 反归一化
（clip [-1,1]、gripper 0.5 二值化、mask），复刻官方 04/26 评测链路。

用法:
  PYTHONPATH=/tmp/starvla_0426 python tools/mowa/wm4a_offline_action_check_oldproto.py \
      --episode 0 --num-probes 12 --port 6695
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

OLD_REPO = Path("/tmp/starvla_0426")
sys.path.insert(0, str(OLD_REPO))

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy  # noqa: E402

REPO = Path("/disk/rl/starVLA")
DATASET = REPO / "playground/Datasets/LEROBOT_LIBERO_DATA/libero_goal_no_noops_1.0.0_lerobot"
STATS_JSON = Path(
    "/disk/rl/models/mowa_init_candidates/WM4A-Wan2d2-OFT-LIBERO-4in1/dataset_statistics.json"
)


def load_episode(episode: int):
    pq_path = DATASET / f"data/chunk-000/episode_{episode:06d}.parquet"
    table = pq.read_table(pq_path, columns=["action", "task_index"])
    actions = np.stack(table.column("action").to_pylist()).astype(np.float32)
    task_index = int(table.column("task_index").to_pylist()[0])
    tasks = [json.loads(line) for line in open(DATASET / "meta/tasks.jsonl")]
    task_desc = next(t["task"] for t in tasks if t["task_index"] == task_index)

    import imageio.v3 as iio

    frames = {}
    for key, cam in (("primary", "observation.images.image"), ("wrist", "observation.images.wrist_image")):
        vid_path = DATASET / f"videos/chunk-000/{cam}/episode_{episode:06d}.mp4"
        frames[key] = iio.imread(vid_path)
    return frames, actions, task_desc


def unnormalize_old(normalized_actions: np.ndarray, stats: dict) -> np.ndarray:
    """旧客户端 unnormalize_actions 的逐行复刻。"""
    mask = np.array(stats.get("mask", np.ones_like(stats["min"])), dtype=bool)
    high, low = np.array(stats["max"]), np.array(stats["min"])
    normalized_actions = np.clip(normalized_actions, -1, 1)
    normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1)
    return np.where(
        mask,
        0.5 * (normalized_actions + 1) * (high - low) + low,
        normalized_actions,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--num-probes", type=int, default=12)
    parser.add_argument("--port", type=int, default=6695)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    frames, actions, task_desc = load_episode(args.episode)
    T = actions.shape[0]
    print(f"episode={args.episode} task={task_desc!r} T={T}")

    stats = json.load(open(STATS_JSON))["franka"]["action"]

    client = WebsocketClientPolicy(args.host, args.port)

    probe_ts = np.linspace(0, T - 9, args.num_probes).astype(int)
    preds, gts = [], []
    for t in probe_ts:
        example = {"image": [frames["primary"][t], frames["wrist"][t]], "lang": task_desc}
        response = client.predict_action({"examples": [example]})
        normalized = np.asarray(response["data"]["normalized_actions"], dtype=np.float32)[0]
        chunk = unnormalize_old(normalized, stats)
        horizon = min(4, chunk.shape[0], T - t)
        preds.append(chunk[:horizon])
        gts.append(actions[t : t + horizon])
        if t == probe_ts[0]:
            print(f"chunk_shape={chunk.shape} first_pred={np.round(chunk[0], 4)}")
            print(f"expert_raw[t]={np.round(actions[t], 4)}")

    pred = np.concatenate(preds)
    gt = np.concatenate(gts)
    mse = ((pred - gt) ** 2).mean(axis=0)
    dims = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    print("\nper-dim  MSE     pred_std  gt_std   corr")
    for i, name in enumerate(dims):
        p, g = pred[:, i], gt[:, i]
        corr = np.corrcoef(p, g)[0, 1] if p.std() > 1e-8 and g.std() > 1e-8 else float("nan")
        print(f"  {name:8s} {mse[i]:.4f}  {p.std():.4f}    {g.std():.4f}   {corr:.3f}")
    print(
        f"\nxyz corr mean: {np.nanmean([np.corrcoef(pred[:, i], gt[:, i])[0, 1] for i in range(3)]):.3f}"
    )


if __name__ == "__main__":
    main()
