"""MoWA E-001 A100 throughput smoke.

This is a smoke-only benchmark for the current P0 FullHeads + action bridge
interface. It does not launch E-001 training, save checkpoints, or modify
checkpoint/resume logic.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from starVLA.model.modules.mowa import (
    MoWAActionBridge,
    MoWAActionBridgeConfig,
    MoWAP0FullHeads,
    MoWAP0FullHeadsConfig,
    build_mowa_p0_constructible_batch_from_smoke,
    mowa_manual_sgd_step,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoWA E-001 A100 throughput smoke.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("playground/Datasets/robocasa365"),
        help="RoboCasa365 dataset root. Default: playground/Datasets/robocasa365",
    )
    parser.add_argument("--batch-candidates", type=str, default="1,2,4,8")
    parser.add_argument("--grad-accum-candidates", type=str, default="1,2,4,8")
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measure-steps", type=int, default=3)
    parser.add_argument("--input-dim", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--action-hidden-dim", type=int, default=2048)
    parser.add_argument("--num-action-layers", type=int, default=36)
    parser.add_argument("--num-bridge-tokens", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs_zh/mowa/mowa_e001_a100_throughput_smoke.json"),
        help="JSON output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_a100_throughput_smoke(args)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


def run_a100_throughput_smoke(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("MoWA E-001 A100 throughput smoke requires CUDA.")

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(device)
    batch_candidates = _parse_positive_ints(args.batch_candidates)
    grad_accum_candidates = _parse_positive_ints(args.grad_accum_candidates)
    max_batch = max(batch_candidates)
    preload_start = time.perf_counter()
    batch = build_mowa_p0_constructible_batch_from_smoke(
        args.data_root,
        max_samples=max_batch,
    )
    preload_sec = time.perf_counter() - preload_start
    if int(batch["metadata"]["sample_count"]) < max_batch:
        raise RuntimeError(
            "MoWA throughput smoke batch source is smaller than requested max batch: "
            f"available={batch['metadata']['sample_count']}, requested={max_batch}."
        )

    torch.manual_seed(0)
    torch.cuda.empty_cache()
    results = []
    for per_device_batch_size in batch_candidates:
        batch_oom = False
        for grad_accum_steps in grad_accum_candidates:
            if batch_oom:
                break
            try:
                result = _measure_candidate(
                    batch=batch,
                    device=device,
                    per_device_batch_size=per_device_batch_size,
                    grad_accum_steps=grad_accum_steps,
                    warmup_steps=args.warmup_steps,
                    measure_steps=args.measure_steps,
                    input_dim=args.input_dim,
                    hidden_dim=args.hidden_dim,
                    action_hidden_dim=args.action_hidden_dim,
                    num_action_layers=args.num_action_layers,
                    num_bridge_tokens=args.num_bridge_tokens,
                    lr=args.lr,
                )
                results.append(result)
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                batch_oom = True
                results.append(
                    {
                        "per_device_batch_size": per_device_batch_size,
                        "gradient_accumulation_steps": grad_accum_steps,
                        "effective_batch_size": per_device_batch_size * grad_accum_steps,
                        "status": "oom",
                        "error": str(exc).split("\n")[0],
                    }
                )

    successful = [item for item in results if item.get("status") == "ok"]
    stable_candidate = _select_stable_candidate(successful)
    return {
        "stage": "P0",
        "experiment_id": "E-001",
        "benchmark": "a100_throughput_smoke",
        "training_started": False,
        "checkpoint_saved": False,
        "gpu": {
            "name": gpu_name,
            "device_count": torch.cuda.device_count(),
            "memory_total_gb": round(
                torch.cuda.get_device_properties(device).total_memory / 1024**3,
                3,
            ),
        },
        "data": {
            "data_root": str(args.data_root),
            "sample_count_loaded": batch["metadata"]["sample_count"],
            "preload_sec": preload_sec,
            "class_mapping_status": batch["metadata"]["class_mapping_status"],
            "source": batch["metadata"]["source"],
        },
        "model": {
            "module": "MoWAP0FullHeads+MoWAActionBridge",
            "input_dim": args.input_dim,
            "hidden_dim": args.hidden_dim,
            "action_hidden_dim": args.action_hidden_dim,
            "num_action_layers": args.num_action_layers,
            "num_bridge_tokens": args.num_bridge_tokens,
            "active_heads": ["task_progress", "action_outcome_class"],
            "masked_heads": [
                "manipulation_readiness",
                "failure_risk",
                "next_best_view_score",
                "subgoal_feasibility",
                "object_visibility_future",
            ],
        },
        "settings": {
            "batch_candidates": batch_candidates,
            "grad_accum_candidates": grad_accum_candidates,
            "warmup_steps": args.warmup_steps,
            "measure_steps": args.measure_steps,
            "precision": "bf16_autocast",
        },
        "results": results,
        "stable_candidate": stable_candidate,
        "go_no_go": (
            "TBD: A100 throughput smoke passed; production E-001 training remains gated"
            if successful
            else "No-Go: no successful A100 throughput candidate"
        ),
        "notes": [
            "Smoke-only measurement; it does not run the full VLA action training stack.",
            "Use results to narrow runtime policy, then validate again with executable E-001 config.",
            "No checkpoint is saved and checkpoint/resume logic is not modified.",
        ],
    }


def _measure_candidate(
    *,
    batch: dict[str, Any],
    device: Any,
    per_device_batch_size: int,
    grad_accum_steps: int,
    warmup_steps: int,
    measure_steps: int,
    input_dim: int,
    hidden_dim: int,
    action_hidden_dim: int,
    num_action_layers: int,
    num_bridge_tokens: int,
    lr: float,
) -> dict[str, Any]:
    import torch

    features = batch["features"][:per_device_batch_size].to(device)
    targets = {
        key: value[:per_device_batch_size].to(device)
        for key, value in batch["targets"].items()
    }
    masks = dict(batch["masks"])
    model = MoWAP0FullHeads(MoWAP0FullHeadsConfig(input_dim=input_dim, hidden_dim=hidden_dim)).to(
        device
    )
    bridge = MoWAActionBridge(
        MoWAActionBridgeConfig(
            wam_feature_dim=hidden_dim,
            action_hidden_dim=action_hidden_dim,
            num_action_layers=num_action_layers,
            num_bridge_tokens=num_bridge_tokens,
        )
    ).to(device)
    model.train()
    bridge.train()
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup_steps):
        _run_optimizer_step(
            model=model,
            bridge=bridge,
            features=features,
            targets=targets,
            masks=masks,
            grad_accum_steps=grad_accum_steps,
            lr=lr,
        )
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    last_loss = None
    for _ in range(measure_steps):
        last_loss = _run_optimizer_step(
            model=model,
            bridge=bridge,
            features=features,
            targets=targets,
            masks=masks,
            grad_accum_steps=grad_accum_steps,
            lr=lr,
        )
    end.record()
    torch.cuda.synchronize(device)
    wall_sec = time.perf_counter() - wall_start
    event_sec = start.elapsed_time(end) / 1000.0
    step_time_sec = event_sec / max(measure_steps, 1)
    effective_batch_size = per_device_batch_size * grad_accum_steps
    return {
        "per_device_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": grad_accum_steps,
        "effective_batch_size": effective_batch_size,
        "status": "ok",
        "optimizer_steps": measure_steps,
        "micro_steps": measure_steps * grad_accum_steps,
        "step_time_sec": step_time_sec,
        "wall_step_time_sec": wall_sec / max(measure_steps, 1),
        "samples_per_sec": effective_batch_size / step_time_sec if step_time_sec > 0 else None,
        "peak_vram_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1024**3,
        "loss": float(last_loss.detach().cpu()) if last_loss is not None else None,
    }


def _run_optimizer_step(
    *,
    model: Any,
    bridge: Any,
    features: Any,
    targets: Mapping[str, Any],
    masks: Mapping[str, Any],
    grad_accum_steps: int,
    lr: float,
) -> Any:
    import torch

    total_loss = None
    for _ in range(grad_accum_steps):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, _, _ = model.compute_loss(features, targets, masks)
            future_features = model.future_features(features, masks)
            bridge_output = bridge(future_features)
            bridge_reg = sum(
                layer_features.float().square().mean()
                for layer_features in bridge_output.layerwise_condition_features
            )
            loss = loss + 1e-6 * bridge_reg
            scaled_loss = loss / grad_accum_steps
        scaled_loss.backward()
        total_loss = loss.detach()
    mowa_manual_sgd_step(model, lr)
    mowa_manual_sgd_step(bridge, lr)
    return total_loss


def _parse_positive_ints(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or any(value < 1 for value in values):
        raise ValueError(f"Expected comma-separated positive integers, got: {raw}")
    return values


def _select_stable_candidate(successful: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not successful:
        return None
    return max(
        successful,
        key=lambda item: (
            item["effective_batch_size"],
            item["samples_per_sec"] or 0.0,
        ),
    )


if __name__ == "__main__":
    main()
