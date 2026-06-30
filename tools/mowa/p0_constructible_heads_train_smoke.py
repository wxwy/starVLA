"""MoWA P0 ConstructibleHeads one-step train smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from starVLA.model.modules.mowa import (
    MoWAP0ConstructibleHeads,
    MoWAP0ConstructibleHeadsConfig,
    build_mowa_p0_constructible_batch_from_smoke,
    mowa_manual_sgd_step,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoWA P0 ConstructibleHeads train smoke.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("playground/Datasets/robocasa365"),
        help="RoboCasa365 dataset root. Default: playground/Datasets/robocasa365",
    )
    parser.add_argument("--input-dim", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. When omitted, prints to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    torch.manual_seed(0)
    batch = build_mowa_p0_constructible_batch_from_smoke(
        args.data_root,
        max_samples=args.max_samples,
    )
    model = MoWAP0ConstructibleHeads(
        MoWAP0ConstructibleHeadsConfig(input_dim=args.input_dim, hidden_dim=args.hidden_dim)
    )
    loss_before, losses_before, _ = model.compute_loss(
        batch["features"],
        batch["targets"],
        batch["masks"],
    )
    loss_before.backward()
    mowa_manual_sgd_step(model, args.lr)
    loss_after, losses_after, outputs = model.compute_loss(
        batch["features"],
        batch["targets"],
        batch["masks"],
    )
    payload = {
        "stage": "P0",
        "module": "MoWAP0ConstructibleHeads",
        "sample_count": batch["metadata"]["sample_count"],
        "constructible_heads": ["task_progress", "action_outcome_class"],
        "masked_heads": [
            "manipulation_readiness",
            "failure_risk",
            "next_best_view_score",
            "subgoal_feasibility",
            "object_visibility_future",
        ],
        "input_shape": list(batch["features"].shape),
        "task_progress_output_shape": list(outputs["task_progress"].shape),
        "action_outcome_class_output_shape": list(outputs["action_outcome_class"].shape),
        "loss_before": float(loss_before.detach().cpu()),
        "loss_after": float(loss_after.detach().cpu()),
        "loss_terms_before": {
            key: float(value.detach().cpu()) for key, value in losses_before.items()
        },
        "loss_terms_after": {
            key: float(value.detach().cpu()) for key, value in losses_after.items()
        },
        "class_mapping_status": batch["metadata"]["class_mapping_status"],
        "go_no_go": "TBD: train smoke passed; production training remains Data Gate",
        "notes": [
            "One optimizer step only; this is not E-001 training.",
            "Features are smoke-only values derived from G0 labels.",
            "Only two ConstructibleHeads are active; other P0 heads remain masked.",
        ],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
