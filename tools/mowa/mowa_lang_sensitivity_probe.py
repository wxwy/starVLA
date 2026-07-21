"""语言敏感度探针：同一观测、切换指令，比较输出动作差异。

对同一帧观测分别用两条指令（默认 "Open the left drawer." / "Open the right
drawer."）请求 predict_action，比较动作 chunk 的 L2 差异与动作自身功率。

判读:
  * relative sensitivity 接近 0 -> 策略几乎忽略语言（instruction-blind）。
  * 明显大于 0 -> 语言条件有效；越大说明指令对动作的调制越强。

Example:
    .venv/bin/python tools/mowa/mowa_lang_sensitivity_probe.py \
        --dataset-dir playground/Datasets/robocasa365/v1.0/target/atomic/OpenDrawer/20250816/lerobot \
        --episodes 4 5 0 1 --payload-style wanpi --port 6791
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "tools" / "mowa"))

from e003_offline_action_check import (  # noqa: E402
    build_history,
    build_parquet_to_training_map,
    decode_video,
    sin_cos_state,
)
from deployment.model_server.tools.websocket_policy_client import (  # noqa: E402
    WebsocketClientPolicy,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--start", type=int, default=20, help="episode 内起始采样帧")
    parser.add_argument("--tail", type=int, default=10, help="episode 末尾保留帧")
    parser.add_argument("--port", type=int, default=6791)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--payload-style", choices=["wanpi", "starflow"], default="wanpi")
    parser.add_argument("--instr-a", default="Open the left drawer.")
    parser.add_argument("--instr-b", default="Open the right drawer.")
    args = parser.parse_args()

    client = WebsocketClientPolicy(args.host, args.port)
    build_parquet_to_training_map(args.dataset_dir)  # 校验 modality.json 可读

    diffs_all, power_all = [], []
    for ep in args.episodes:
        df = pd.read_parquet(args.dataset_dir / "data" / "chunk-000" / f"episode_{ep:06d}.parquet")
        T = len(df)
        vids = {}
        for cam in ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"):
            vp = args.dataset_dir / "videos" / "chunk-000" / f"observation.images.{cam}" / f"episode_{ep:06d}.mp4"
            vids[cam] = decode_video(vp) if vp.exists() else None

        def make_example(t: int, instruction: str):
            if args.payload_style == "wanpi":
                return {
                    "mowa_multi_view_images": [
                        build_history(vids["robot0_agentview_left"], t),
                        build_history(vids["robot0_eye_in_hand"], t),
                    ],
                    "lang": instruction,
                    "state": sin_cos_state(df["observation.state"].iloc[t]),
                }
            return {
                "image": [
                    vids["robot0_agentview_left"][t],
                    vids["robot0_agentview_right"][t],
                    vids["robot0_eye_in_hand"][t],
                ],
                "lang": instruction,
                "state": sin_cos_state(df["observation.state"].iloc[t]),
            }

        for t in range(args.start, T - args.tail, args.stride):
            preds = {}
            for instr in (args.instr_a, args.instr_b):
                resp = client.predict_action({
                    "examples": [make_example(t, instr)],
                    "do_sample": False, "use_ddim": True, "num_ddim_steps": 10,
                    "unnorm_key": None,
                })
                preds[instr] = np.asarray(resp["data"]["actions"], dtype=np.float64)[0, :8]
            d = float(np.mean((preds[args.instr_a] - preds[args.instr_b]) ** 2))
            scale = float(np.mean(preds[args.instr_a] ** 2 + preds[args.instr_b] ** 2)) / 2
            diffs_all.append(d)
            power_all.append(scale)
            print(f"ep{ep} t={t:3d}: L2(A,B)={d:.5f}  mean_pred_power={scale:.5f}")

    print(f"\n=== {len(diffs_all)} samples ({args.payload_style}) ===")
    print(f"mean L2(A, B)        = {np.mean(diffs_all):.5f}")
    print(f"mean pred power      = {np.mean(power_all):.5f}")
    print(f"relative sensitivity = {np.mean(diffs_all) / max(np.mean(power_all), 1e-9):.4f}")
    print("(relative sensitivity << 1  =>  输出几乎不随指令变化，疑似 instruction-blind)")


if __name__ == "__main__":
    main()
