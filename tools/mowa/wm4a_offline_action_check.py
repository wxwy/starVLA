"""WM4A-WanOFT-LIBERO-4in1 离线动作对照。

把 libero_goal 专家轨迹的观测（primary/wrist 视频帧 + 任务指令）逐帧发给
正在运行的 policy server，比较返回的 unnorm 动作块与专家动作。

用法:
  python tools/mowa/wm4a_offline_action_check.py \
      --episode 0 --num-probes 12 --port 6694
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy  # noqa: E402

DATASET = REPO / "playground/Datasets/LEROBOT_LIBERO_DATA/libero_goal_no_noops_1.0.0_lerobot"


def load_episode(episode: int) -> tuple[dict, np.ndarray, str]:
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
        frames[key] = iio.imread(vid_path)  # [T,H,W,3] uint8
    return frames, actions, task_desc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--num-probes", type=int, default=12)
    parser.add_argument("--port", type=int, default=6694)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    frames, actions, task_desc = load_episode(args.episode)
    T = actions.shape[0]
    print(f"episode={args.episode} task={task_desc!r} T={T} action_shape={actions.shape}")
    print(f"primary_frames={frames['primary'].shape} wrist_frames={frames['wrist'].shape}")

    client = WebsocketClientPolicy(args.host, args.port)
    meta = client.get_server_metadata()
    print("server metadata:", json.dumps(meta, default=str)[:400])

    probe_ts = np.linspace(0, T - 9, args.num_probes).astype(int)
    preds, gts = [], []
    for t in probe_ts:
        example = {
            "image": [frames["primary"][t], frames["wrist"][t]],
            "lang": task_desc,
        }
        response = client.predict_action({"examples": [example], "do_sample": False})
        chunk = np.asarray(response["data"]["actions"], dtype=np.float32)[0]  # [H,7]
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
    print(f"\nxyz corr mean: {np.nanmean([np.corrcoef(pred[:,i], gt[:,i])[0,1] for i in range(3)]):.3f}")


if __name__ == "__main__":
    main()
