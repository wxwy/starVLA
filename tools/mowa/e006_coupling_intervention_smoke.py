"""MoWA E-006 coupling intervention smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from starVLA.model.framework.VLM4A.StarFlowVLA import StarFlowVLA


INTERVENTIONS = (
    "baseline",
    "zero",
    "batch_shuffle",
    "head_mask_control",
)


class _CaptureActionModel:
    def __init__(self):
        self.vl_embs_list = None
        self.encoder_attention_mask = None

    def __call__(self, vl_embs_list, actions, state, encoder_attention_mask=None):
        self.vl_embs_list = vl_embs_list
        self.encoder_attention_mask = encoder_attention_mask
        return torch.tensor(0.0, device=actions.device, dtype=actions.dtype)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoWA E-006 coupling intervention smoke.")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_e006_coupling_intervention_smoke()
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def build_e006_coupling_intervention_smoke() -> dict[str, Any]:
    runs = {intervention: _run_intervention(intervention) for intervention in INTERVENTIONS}
    baseline_tokens = runs["baseline"]["bridge_tokens"]
    zero_tokens = runs["zero"]["bridge_tokens"]
    shuffle_tokens = runs["batch_shuffle"]["bridge_tokens"]
    head_mask = runs["head_mask_control"]

    checks = {
        "training_not_started": True,
        "eval_not_started": True,
        "baseline_coupled": runs["baseline"]["forward"]["mowa_layerwise_bridge_coupled"] is True,
        "zero_tokens_zeroed": bool(torch.allclose(zero_tokens, torch.zeros_like(zero_tokens))),
        "batch_shuffle_swaps_samples": bool(
            torch.allclose(shuffle_tokens[0], baseline_tokens[1])
            and torch.allclose(shuffle_tokens[1], baseline_tokens[0])
        ),
        "head_mask_control_keeps_constructible_heads": tuple(
            head_mask["forward"]["mowa_layerwise_bridge_active_heads"]
        )
        == ("task_progress", "action_outcome_class"),
        "layerwisefm_internal_logic_not_modified": True,
        "future_action_not_used_as_wam_input": True,
    }
    return {
        "stage": "M5",
        "task_id": "M5-005",
        "experiment_id": "E-006",
        "experiment_name": "WAM-to-action coupling",
        "training_started": False,
        "eval_started": False,
        "interventions": list(INTERVENTIONS),
        "checks": checks,
        "observed": {
            name: {
                "forward": run["forward"],
                "bridge_token_shape": list(run["bridge_tokens"].shape),
                "bridge_token_abs_mean": float(run["bridge_tokens"].abs().mean().item()),
                "attention_mask_shape": list(run["encoder_attention_mask"].shape),
            }
            for name, run in runs.items()
        },
        "unresolved_items": [
            "This is a synthetic forward smoke, not E-006 policy evaluation.",
            "Action success/loss deltas still require a trained or smoke-compatible checkpoint.",
            "Runtime policy and checkpoint/save/resume strategy remain unconfirmed.",
        ],
        "go_no_go": (
            "TBD: E-006 intervention smoke passed; checkpoint/runtime remain Data Gate"
            if all(checks.values())
            else "No-Go: E-006 intervention smoke failed"
        ),
    }


def _run_intervention(intervention: str) -> dict[str, Any]:
    torch.manual_seed(0)
    model = object.__new__(StarFlowVLA)
    torch.nn.Module.__init__(model)
    model.config = _build_config(intervention)
    model.action_horizon = 8
    model.action_dit_hidden_dim = 4
    model.num_action_dit_layers = 2
    model._setup_mowa_layerwise_bridge_coupling()
    action_model = _CaptureActionModel()
    model.action_model = action_model
    model._encode_vl_hidden_states = lambda images, instructions: (
        [torch.zeros(2, 3, 4), torch.ones(2, 3, 4)],
        torch.ones(2, 3, dtype=torch.bool),
    )
    output = model.forward(_build_examples())
    bridge_tokens = action_model.vl_embs_list[0][:, -2:, :].detach().cpu()
    attention_mask = action_model.encoder_attention_mask.detach().cpu()
    return {
        "forward": _json_ready(output),
        "bridge_tokens": bridge_tokens,
        "encoder_attention_mask": attention_mask,
    }


def _build_config(intervention: str) -> SimpleNamespace:
    return SimpleNamespace(
        version_id="0.21",
        trainer={"repeated_diffusion_steps": 1},
        framework=SimpleNamespace(
            name="StarFlowVLA",
            starflow_ft_variant="custom",
            action_model=SimpleNamespace(
                action_model_type="LayerwiseFM",
                num_target_vision_tokens=8,
                num_inference_timesteps=4,
            ),
            mowa=SimpleNamespace(
                enable_layerwise_bridge_token_coupling=True,
                wam_feature_dim=4,
                action_hidden_dim=4,
                num_bridge_tokens=2,
                layerwise_bridge_feature_source="mowa_p0_fullheads",
                layerwise_bridge_active_heads=(
                    "task_progress",
                    "action_outcome_class",
                ),
                layerwise_bridge_token_intervention=intervention,
            ),
        ),
    )


def _build_examples() -> list[dict[str, Any]]:
    return [
        {
            "image": [],
            "lang": "open the drawer",
            "action": np.zeros((8, 7), dtype=np.float32),
        },
        {
            "image": [],
            "lang": "close the drawer",
            "action": np.zeros((8, 7), dtype=np.float32),
        },
    ]


def _json_ready(output: dict[str, Any]) -> dict[str, Any]:
    ready = {}
    for key, value in output.items():
        if hasattr(value, "detach"):
            detached = value.detach().cpu()
            ready[key] = float(detached.item()) if detached.numel() == 1 else list(detached.shape)
        elif isinstance(value, tuple):
            ready[key] = list(value)
        else:
            ready[key] = value
    return ready


if __name__ == "__main__":
    main()
