#!/usr/bin/env python3
"""E003-B1 checkpoint 推理工具，兼容 E003/E003-B2 原双视角 GPU smoke。"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from starVLA.dataloader.mowa.latent_cache_dataset import MoWALatentCacheDataset
from starVLA.model.framework.WM4A.WanPI import Wan_PI
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils


DEFAULT_CONFIG = Path(
    "configs/mowa/mowa_e003_future_latent_prior_long_training_launch_candidate.yaml"
)
DEFAULT_CACHE_ROOT = Path(
    "playground/Datasets/robocasa365_wan2.2_latent/v1.0/target/atomic"
)
DEFAULT_MAIN_VIEW = "observation.images.robot0_agentview_left"
DEFAULT_WRIST_VIEW = "observation.images.robot0_eye_in_hand"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--text-cache", type=Path, default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="可选 lightweight checkpoint；在构建 WanPI 后加载其可训练参数。",
    )
    parser.add_argument("--history-steps", type=int, default=1)
    parser.add_argument("--future-steps", type=int, default=8)
    parser.add_argument("--action-steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--optimizer", choices=("sgd", "adamw"), default="sgd")
    parser.add_argument("--monitor-interval", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="可选 JSON 报告路径；仅 checkpoint 推理模式写入结果。",
    )
    parser.add_argument("--validate-data-flow", action="store_true")
    parser.add_argument("--validation-steps", type=int, default=2)
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="跳过训练步骤，仅加载 checkpoint（若指定）并执行一次双视角 predict_action。",
    )
    parser.add_argument(
        "--run-inference",
        action="store_true",
        help="训练 smoke 后用同一批 history/current latent 执行一次双视角 future-flow 推理。",
    )
    parser.add_argument(
        "--run-online-inference",
        action="store_true",
        help="以原始双视角帧历史触发在线 Wan VAE/text cache 推理。",
    )
    return parser.parse_args()


def _gpu_monitor(stop_event: threading.Event, interval: float) -> None:
    command = (
        "nvidia-smi",
        "--query-gpu=timestamp,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    )
    while not stop_event.wait(interval):
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"\n[gpu] {result.stdout.strip()}", flush=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            print(f"\n[gpu] monitor stopped: {exc}", flush=True)
            return


def _memory_gib(value: int) -> float:
    return value / 2**30


def _write_inference_report(path: Path | None, report: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[smoke] inference report: {path}", flush=True)


def _build_inference_report(
    *,
    args: argparse.Namespace,
    prediction: dict,
    elapsed_sec: float,
) -> dict:
    actions = prediction["normalized_actions"]
    future_latents = prediction["mowa_predicted_future_latents"]
    done_logits = prediction["mowa_future_done_logits"]
    checks = {
        "action_batch_matches": actions.shape[0] == args.batch_size,
        "action_horizon_matches": actions.shape[1] == args.action_steps,
        "future_batch_matches": future_latents.shape[0] == args.batch_size,
        "future_view_count_matches": future_latents.shape[1] == 2,
        "future_horizon_matches": future_latents.shape[3] == args.future_steps,
        "done_shape_matches": tuple(done_logits.shape) == (args.batch_size, args.future_steps),
        "actions_finite": bool(torch.isfinite(actions).all()),
        "future_latents_finite": bool(torch.isfinite(future_latents).all()),
        "done_logits_finite": bool(torch.isfinite(done_logits).all()),
    }
    return {
        "stage": "E003-B1 checkpoint inference",
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        "history_steps": args.history_steps,
        "future_steps": args.future_steps,
        "action_steps": args.action_steps,
        "batch_size": args.batch_size,
        "data_flow_validation_enabled": args.validate_data_flow,
        "prediction": {
            "actions_shape": list(actions.shape),
            "future_latents_shape": list(future_latents.shape),
            "done_logits_shape": list(done_logits.shape),
            "flow_steps": int(prediction["mowa_future_flow_steps"]),
            "elapsed_sec": elapsed_sec,
            "peak_allocated_gib": _memory_gib(torch.cuda.max_memory_allocated()),
            "peak_reserved_gib": _memory_gib(torch.cuda.max_memory_reserved()),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    args = parse_args()
    if (
        (not args.inference_only and args.steps <= 0)
        or args.batch_size <= 0
        or args.history_steps < 0
    ):
        raise ValueError(
            "batch_size must be positive, history_steps must be non-negative, and "
            "steps must be positive unless --inference-only is set."
        )
    if args.checkpoint is not None and not args.checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {args.checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke.")

    manifest = args.manifest or (
        args.cache_root / "window_manifests/wan2.2_h10_f8_train.parquet"
    )
    text_cache = args.text_cache or (args.cache_root / "instruction_text_latents.pt")
    print("[smoke] configuration", flush=True)
    print(f"  config: {args.config}", flush=True)
    print(f"  manifest: {manifest}", flush=True)
    print(f"  text_cache: {text_cache}", flush=True)
    print(f"  checkpoint: {args.checkpoint}", flush=True)
    print(
        f"  history/future/action: {args.history_steps}/{args.future_steps}/{args.action_steps}",
        flush=True,
    )
    print(
        f"  steps: {args.steps} | batch_size: {args.batch_size} | "
        f"optimizer: {args.optimizer} | device: {args.device}",
        flush=True,
    )
    print(
        f"  inference_only: {args.inference_only} | run_inference: {args.run_inference} | "
        f"run_online_inference: {args.run_online_inference}",
        flush=True,
    )

    dataset = MoWALatentCacheDataset(
        args.cache_root,
        manifest_path=manifest,
        instruction_text_latent=text_cache,
        history_steps=args.history_steps,
        future_steps=args.future_steps,
        action_chunk_steps=args.action_steps,
        video_keys=(DEFAULT_MAIN_VIEW, DEFAULT_WRIST_VIEW),
    )
    keys = dataset.sample_keys[: args.batch_size]
    if len(keys) != args.batch_size:
        raise ValueError(f"Dataset only provides {len(keys)} samples for batch_size={args.batch_size}.")
    samples = []
    for key in keys:
        sample = dataset.get_sample(
            episode_index=key[0],
            anchor_index=key[1],
            video_key=key[2],
        )
        if not args.inference_only:
            sample["action"] = np.zeros((args.action_steps, 12), dtype=np.float32)
            sample["state"] = np.zeros((1, 32), dtype=np.float32)
        samples.append(sample)
    sample = samples[0]
    print(
        "[smoke] sample shapes "
        f"history={tuple(sample['mowa_multi_view_history_latents'].shape)} "
        f"current={tuple(sample['mowa_multi_view_current_latents'].shape)} "
        f"future={tuple(sample['mowa_multi_view_future_latents'].shape)}",
        flush=True,
    )

    config = OmegaConf.load(args.config)
    config.latent_cache.history_window_steps = args.history_steps
    config.framework.mowa.validate_data_flow = args.validate_data_flow
    config.framework.mowa.validation_steps = args.validation_steps
    model = Wan_PI(config).to(args.device)
    if args.checkpoint is not None:
        TrainerUtils.load_pretrained_backbones(
            model,
            str(args.checkpoint),
            preferred_format="safetensors",
        )
        print(f"[smoke] checkpoint loaded: {args.checkpoint}", flush=True)
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(False)
    torch.cuda.reset_peak_memory_stats()
    if not args.inference_only:
        model.train()
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
        else:
            optimizer = torch.optim.SGD(trainable, lr=args.learning_rate)

        stop_event = threading.Event()
        monitor = threading.Thread(
            target=_gpu_monitor,
            args=(stop_event, args.monitor_interval),
            daemon=True,
        )
        monitor.start()
        progress = tqdm(range(args.steps), desc="GPU smoke", unit="step", dynamic_ncols=True)
        try:
            for step in progress:
                step_start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                output = model(samples)
                action_loss = output["action_loss"].mean()
                future_loss = output["mowa_future_latent_prior_loss"]
                latent_loss = output["loss_future_total"]
                done_loss = output["loss_done"]
                loss = action_loss + future_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at step {step}: {loss}.")
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
                progress.set_postfix(
                    loss=f"{float(loss.detach()):.4f}",
                    latent=f"{float(latent_loss.detach()):.4f}",
                    done=f"{float(done_loss.detach()):.4f}",
                    action=f"{float(action_loss.detach()):.4f}",
                    step_s=f"{time.perf_counter() - step_start:.2f}",
                    peak_gib=f"{_memory_gib(torch.cuda.max_memory_allocated()):.2f}",
                )
        finally:
            progress.close()
            stop_event.set()
            monitor.join(timeout=max(1.0, args.monitor_interval * 2))

    print("[smoke] finished", flush=True)
    print(
        f"  peak_allocated: {_memory_gib(torch.cuda.max_memory_allocated()):.3f} GiB",
        flush=True,
    )
    print(
        f"  peak_reserved: {_memory_gib(torch.cuda.max_memory_reserved()):.3f} GiB",
        flush=True,
    )
    if args.run_inference or args.inference_only:
        # 明确删除训练专用 future GT/action, 验证 predict_action 不会静默读取它们。
        inference_samples = [
            {
                key: value
                for key, value in item.items()
                if key not in {"mowa_multi_view_future_latents", "action", "mowa_future_done_target"}
            }
            for item in samples
        ]
        model.eval()
        inference_start = time.perf_counter()
        with torch.inference_mode():
            prediction = model.predict_action(inference_samples)
        torch.cuda.synchronize()
        inference_elapsed = time.perf_counter() - inference_start
        print("[smoke] dual-view inference finished", flush=True)
        print(
            f"  actions: {tuple(prediction['normalized_actions'].shape)} | "
            f"future_latents: {tuple(prediction['mowa_predicted_future_latents'].shape)} | "
            f"done_logits: {tuple(prediction['mowa_future_done_logits'].shape)} | "
            f"flow_steps: {prediction['mowa_future_flow_steps']} | "
            f"elapsed: {inference_elapsed:.2f}s",
            flush=True,
        )
        if args.inference_only:
            report = _build_inference_report(
                args=args,
                prediction=prediction,
                elapsed_sec=inference_elapsed,
            )
            _write_inference_report(args.output, report)
            if not report["passed"]:
                raise RuntimeError(f"E003-B1 checkpoint inference checks failed: {report['checks']}")
    if args.run_online_inference:
        if args.history_steps != 0:
            raise ValueError("--run-online-inference currently requires --history-steps 0.")
        online_samples = []
        raw_frames = [np.zeros((256, 256, 3), dtype=np.uint8) for _ in range(5)]
        for item in samples:
            online_samples.append(
                {
                    key: value
                    for key, value in item.items()
                    if key
                    not in {
                        "mowa_multi_view_history_latents",
                        "mowa_multi_view_current_latents",
                        "mowa_multi_view_future_latents",
                        "mowa_future_done_target",
                        "text_embeds",
                        "text_attention_mask",
                        "action",
                    }
                }
            )
            online_samples[-1]["mowa_multi_view_images"] = [list(raw_frames), list(raw_frames)]
        model.eval()
        online_start = time.perf_counter()
        prediction = model.predict_action(online_samples)
        torch.cuda.synchronize()
        print("[smoke] online dual-view inference finished", flush=True)
        print(
            f"  actions: {tuple(prediction['normalized_actions'].shape)} | "
            f"future_latents: {tuple(prediction['mowa_predicted_future_latents'].shape)} | "
            f"done_logits: {tuple(prediction['mowa_future_done_logits'].shape)} | "
            f"elapsed: {time.perf_counter() - online_start:.2f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
