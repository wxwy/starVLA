"""Offline action-consistency check for MoWA-E-003 style policy servers.

Replays real training-dataset episodes through the *eval-time* server path
(websocket policy server + PolicyWarper payload format) and compares the
predicted action chunks against the expert actions stored in the dataset.

Purpose: bisect "0% sim success" failures.
  * Offline error small  -> model fits the data; the problem lives on the
    sim/observation side (camera pose, state definition, instruction, ...).
  * Offline error large  -> the model / training / unnorm path itself is off.

The payload construction mirrors
``examples/Robocasa_365/eval_files/model2robocasa365_interface.py``:
  * 5-frame Wan history per view, front-padded with the earliest frame.
  * state = sin/cos(raw 16-D state) -> (1, 32).
  * instruction from the episode's ``annotation.human.task_description``.

Since the training data config normalizes actions with min_max and the
dataset statistics have min=-1 / max=+1 per dim, the server's unapply is an
identity map — server outputs are directly comparable to dataset actions.

Example:
    python tools/mowa/e003_offline_action_check.py \
        --dataset-dir playground/Datasets/robocasa365/v1.0/target/atomic/OpenDrawer/20250816/lerobot \
        --ckpt-path playground/mowa_ckpt/MoWA-E-003_future_latent_prior_wo_history_lora_260718_0039/checkpoints/steps_28500 \
        --episodes 0 1 2 --stride 15 --port 6791 \
        --output /tmp/e003_offline_check.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import av
import numpy as np
import pandas as pd

from deployment.model_server.tools.websocket_policy_client import (
    WebsocketClientPolicy,
)

WAN_HISTORY_FRAMES = 5
ACTION_DIM = 12
ACTION_CHUNK = 32
ACTION_KEY_SPLITS = {
    "eef_pos": (0, 3),
    "eef_rot": (3, 6),
    "gripper_close": (6, 7),
    "base_motion": (7, 11),
    "control_mode": (11, 12),
}


def decode_video(video_path: Path) -> np.ndarray:
    """Decode an mp4 to (T, H, W, 3) uint8 RGB frames."""
    frames = []
    with av.open(str(video_path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    return np.stack(frames, axis=0)


def build_history(frames: np.ndarray, t: int) -> list[np.ndarray]:
    """Last 5 consecutive frames ending at t, front-padded with the earliest."""
    start = max(0, t - WAN_HISTORY_FRAMES + 1)
    hist = [frames[i] for i in range(start, t + 1)]
    return [hist[0]] * (WAN_HISTORY_FRAMES - len(hist)) + hist


def sin_cos_state(state: np.ndarray) -> np.ndarray:
    """(16,) -> (1, 32), matching the eval client / training transform."""
    state = np.asarray(state, dtype=np.float64)
    return np.concatenate([np.sin(state), np.cos(state)], axis=-1)[None, :]


def load_tasks(dataset_dir: Path) -> dict[int, str]:
    tasks = {}
    with (dataset_dir / "meta" / "tasks.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            tasks[int(row["task_index"])] = row["task"]
    return tasks


# Training-time action layout (DataConfig.action_keys order, also the order the
# policy server returns). The parquet `action` column stores a DIFFERENT order
# (see meta/modality.json), so expert actions must be remapped before compare.
TRAINING_ACTION_LAYOUT = [
    ("end_effector_position", 3),
    ("end_effector_rotation", 3),
    ("gripper_close", 1),
    ("base_motion", 4),
    ("control_mode", 1),
]


def build_parquet_to_training_map(dataset_dir: Path) -> np.ndarray:
    """Return indices perm such that action_training = action_parquet[..., perm]."""
    modality = json.load((dataset_dir / "meta" / "modality.json").open())
    parquet_slices = modality["action"]
    perm: list[int] = []
    for key, dim in TRAINING_ACTION_LAYOUT:
        sl = parquet_slices[key]
        assert sl["end"] - sl["start"] == dim, f"{key}: modality dim mismatch"
        perm.extend(range(sl["start"], sl["end"]))
    return np.asarray(perm, dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--ckpt-path", required=True, type=Path,
                        help="Used only for sanity-checking dataset_statistics.json")
    parser.add_argument("--episodes", type=int, nargs="+", default=[0])
    parser.add_argument("--stride", type=int, default=15,
                        help="Sample every `stride` frames within an episode")
    parser.add_argument("--compare-horizon", type=int, default=32,
                        help="Only compare the first N steps of each predicted chunk "
                             "(eval executes 8 before re-planning)")
    parser.add_argument("--dump-examples", type=int, default=0,
                        help="Save this many pred/expert chunk trajectories into the report")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6791)
    parser.add_argument(
        "--payload-style",
        choices=["wanpi", "starflow"],
        default="wanpi",
        help="wanpi: mowa_multi_view_images with 5-frame history (WanPI/E-003). "
             "starflow: image=[agentview_left, agentview_right, eye_in_hand] single "
             "current frame per view (StarFlowVLA/E-001 baseline).",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    # --- sanity: min_max unnorm must be ~identity for comparability ---------
    stats_path = args.ckpt_path.parent.parent / "dataset_statistics.json"
    stats = json.load(stats_path.open())
    emb = next(iter(stats))
    a_min = np.array(stats[emb]["action"]["min"], dtype=float)
    a_max = np.array(stats[emb]["action"]["max"], dtype=float)
    print(f"[check] unnorm_key={emb} action min range [{a_min.min():.3f},{a_min.max():.3f}]"
          f" max range [{a_max.min():.3f},{a_max.max():.3f}]")
    if not (np.allclose(a_min, -1.0, atol=0.05) and np.allclose(a_max, 1.0, atol=0.05)):
        print("[warn] action stats are not ±1 — server unapply is NOT identity; "
              "comparison still valid only if dataset stores raw actions.")

    tasks = load_tasks(args.dataset_dir)
    action_perm = build_parquet_to_training_map(args.dataset_dir)
    client = WebsocketClientPolicy(args.host, args.port)
    meta = client.get_server_metadata()
    print(f"[check] server meta: ckpt={meta.get('ckpt_path')} "
          f"action_chunk_size={meta.get('action_chunk_size')}")

    all_pred, all_expert = [], []
    per_sample_records = []
    dumped_examples: list[dict] = []
    t_start = time.time()

    for ep in args.episodes:
        parquet = args.dataset_dir / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"
        df = pd.read_parquet(parquet)
        T = len(df)
        desc_idx = int(df["annotation.human.task_description"].iloc[0])
        instruction = tasks[desc_idx]

        videos = {}
        cam_map = {
            "wanpi": (("main", "robot0_agentview_left"), ("wrist", "robot0_eye_in_hand")),
            "starflow": (
                ("main", "robot0_agentview_left"),
                ("right", "robot0_agentview_right"),
                ("wrist", "robot0_eye_in_hand"),
            ),
        }[args.payload_style]
        for view, cam in cam_map:
            vp = args.dataset_dir / "videos" / "chunk-000" / f"observation.images.{cam}" / f"episode_{ep:06d}.mp4"
            if not vp.exists():
                raise FileNotFoundError(vp)
            videos[view] = decode_video(vp)
            assert len(videos[view]) == T, f"{view} frames {len(videos[view])} != parquet rows {T}"

        for t in range(0, T, args.stride):
            horizon = min(ACTION_CHUNK, T - t, args.compare_horizon)
            if horizon < 2:
                continue
            if args.payload_style == "wanpi":
                example = {
                    "mowa_multi_view_images": [
                        build_history(videos["main"], t),
                        build_history(videos["wrist"], t),
                    ],
                    "lang": instruction,
                    "state": sin_cos_state(df["observation.state"].iloc[t]),
                }
            else:
                example = {
                    "image": [videos["main"][t], videos["right"][t], videos["wrist"][t]],
                    "lang": instruction,
                    "state": sin_cos_state(df["observation.state"].iloc[t]),
                }
            vla_input = {
                "examples": [example],
                "do_sample": False,
                "use_ddim": True,
                "num_ddim_steps": 10,
                "unnorm_key": None,
            }
            resp = client.predict_action(vla_input)
            pred = np.asarray(resp["data"]["actions"], dtype=np.float64)[0, :horizon]
            expert = np.stack(df["action"].iloc[t:t + horizon].to_numpy()).astype(np.float64)
            expert = expert[:, action_perm]  # parquet layout -> training layout

            all_pred.append(pred)
            all_expert.append(expert)
            per_sample_records.append({
                "episode": ep, "t": t, "horizon": horizon,
                "mse": float(np.mean((pred - expert) ** 2)),
                "pred_std": float(pred.std()),
                "expert_std": float(expert.std()),
            })
            if len(dumped_examples) < args.dump_examples:
                dumped_examples.append({
                    "episode": ep, "t": t,
                    "pred": np.round(pred, 4).tolist(),
                    "expert": np.round(expert, 4).tolist(),
                })
        print(f"[check] episode {ep} ({instruction!r}): "
              f"{len([r for r in per_sample_records if r['episode'] == ep])} samples done")

    elapsed = time.time() - t_start
    P = np.concatenate(all_pred, axis=0)     # (N, 12)
    E = np.concatenate(all_expert, axis=0)   # (N, 12)
    n = len(P)

    zero_mse = float(np.mean(E ** 2))
    overall_mse = float(np.mean((P - E) ** 2))
    overall_mae = float(np.mean(np.abs(P - E)))

    per_dim = {}
    for d in range(ACTION_DIM):
        err = P[:, d] - E[:, d]
        var = float(E[:, d].var())
        per_dim[d] = {
            "mse": float(np.mean(err ** 2)),
            "mae": float(np.mean(np.abs(err))),
            "expert_var": var,
            "nmse": float(np.mean(err ** 2) / var) if var > 1e-12 else None,
            "corr": float(np.corrcoef(P[:, d], E[:, d])[0, 1]) if var > 1e-12 else None,
            "pred_std": float(P[:, d].std()),
            "expert_std": float(E[:, d].std()),
        }

    per_key = {}
    for key, (s, e) in ACTION_KEY_SPLITS.items():
        dims = list(range(s, e))
        mses = [per_dim[d]["mse"] for d in dims]
        corrs = [per_dim[d]["corr"] for d in dims if per_dim[d]["corr"] is not None]
        per_key[key] = {
            "mse": float(np.mean(mses)),
            "mean_corr": float(np.mean(corrs)) if corrs else None,
        }

    report = {
        "dataset_dir": str(args.dataset_dir),
        "episodes": args.episodes,
        "stride": args.stride,
        "n_samples": len(per_sample_records),
        "n_action_steps_compared": n,
        "elapsed_sec": elapsed,
        "overall": {
            "model_mse": overall_mse,
            "model_mae": overall_mae,
            "zero_baseline_mse": zero_mse,
            "mse_ratio_vs_zero": overall_mse / zero_mse if zero_mse > 0 else None,
        },
        "per_key": per_key,
        "per_dim": per_dim,
        "compare_horizon": args.compare_horizon,
        "examples": dumped_examples,
        "samples": per_sample_records,
    }

    print(f"\n==== offline action check ({n} action steps from "
          f"{len(per_sample_records)} predictions, {elapsed:.0f}s) ====")
    print(f"model MSE        : {overall_mse:.4f}")
    print(f"zero-baseline MSE: {zero_mse:.4f}")
    print(f"ratio            : {overall_mse / zero_mse:.3f}  (<1 means better than predicting zero)")
    for key, v in per_key.items():
        corr = f"{v['mean_corr']:.3f}" if v['mean_corr'] is not None else "n/a"
        print(f"  {key:14s} mse={v['mse']:.4f}  mean_corr={corr}")
    print("\nper-dim detail (mse / nmse / corr / pred_std vs expert_std):")
    for d, v in per_dim.items():
        nmse = f"{v['nmse']:.3f}" if v['nmse'] is not None else " n/a "
        corr = f"{v['corr']:.3f}" if v['corr'] is not None else " n/a "
        print(f"  dim{d:2d}: mse={v['mse']:.4f} nmse={nmse} corr={corr} "
              f"std={v['pred_std']:.3f}/{v['expert_std']:.3f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print(f"\nreport written to {args.output}")


if __name__ == "__main__":
    main()
