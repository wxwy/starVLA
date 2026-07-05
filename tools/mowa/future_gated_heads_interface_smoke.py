#!/usr/bin/env python3
"""MoWA M2-003 Future-GatedHeads interface smoke.

Validates that the gated-head wrapper produces gate values in [0, 1],
maintains shape compatibility with FullHeads, and logs gate values.
No training is started.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from starVLA.model.modules.mowa.gated_heads import MoWAGatedHeads, MoWAGatedHeadsConfig
from starVLA.model.modules.mowa.p0_heads import MoWAFutureFullHeadsConfig
from starVLA.mowa_constants import MOWA_FUTURE_FULL_HEADS, MOWA_FUTURE_CONSTRUCTIBLE_HEADS

OUTPUT = Path("docs_zh/mowa/mowa_future_gated_heads_interface_smoke.json")


def _build_smoke() -> dict[str, Any]:
    cfg = MoWAGatedHeadsConfig(
        heads_config=MoWAFutureFullHeadsConfig(
            input_dim=1024,
            hidden_dim=512,
        ),
        init_gate=0.5,
    )
    model = MoWAGatedHeads(cfg)

    # --- gate shape / range check ---
    gate_dict = model.gate_values()
    gates_ok = (
        len(gate_dict) == len(MOWA_FUTURE_FULL_HEADS)
        and all(0.0 <= v <= 1.0 for v in gate_dict.values())
    )

    # --- forward smoke ---
    batch, dim = 4, 1024
    x = torch.randn(batch, dim)
    masks = {h: h in MOWA_FUTURE_CONSTRUCTIBLE_HEADS for h in MOWA_FUTURE_FULL_HEADS}
    out = model(x, masks)

    forward_ok = (
        out.hidden_features.shape == (batch, 512)
        and len(out.head_outputs) == len(MOWA_FUTURE_FULL_HEADS)
        and set(out.active_heads) == set(MOWA_FUTURE_CONSTRUCTIBLE_HEADS)
    )

    all_ok = gates_ok and forward_ok
    return {
        "task_id": "M2-003",
        "experiment_id": "E-002",
        "training_started": False,
        "checks": {
            "gate_shape_matches_head_count": len(gate_dict) == len(MOWA_FUTURE_FULL_HEADS),
            "gate_values_in_range": all(0.0 <= v <= 1.0 for v in gate_dict.values()),
            "forward_shape_ok": forward_ok,
            "head_names_match_fullheads": set(model.head_names) == set(MOWA_FUTURE_FULL_HEADS),
        },
        "observed": {
            "gate_values": {k: round(v, 6) for k, v in gate_dict.items()},
            "hidden_features_shape": list(out.hidden_features.shape),
            "active_heads": list(out.active_heads),
            "masked_heads": list(out.masked_heads),
        },
        "go_no_go": "TBD: gated-heads interface smoke passed" if all_ok else "No-Go: gate/forward check failed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    payload = _build_smoke()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"go_no_go: {payload['go_no_go']}")


if __name__ == "__main__":
    main()
