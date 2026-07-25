#!/usr/bin/env python3
"""MoWA M6-001 frozen decoder diagnostic plan smoke.

Validates that the diagnostic plan YAML exists and records its
constraints.  Does NOT run a Wan decoder, VAE, or video generation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

PLAN_YAML = Path("configs/mowa/mowa_frozen_decoder_diagnostic_plan.yaml")
OUTPUT = Path("docs_zh/mowa/mowa_frozen_decoder_diagnostic_smoke.json")


def _build_smoke() -> dict[str, Any]:
    if not PLAN_YAML.is_file():
        return {
            "task_id": "M6-001",
            "experiment_id": "E-008",
            "eval_started": False,
            "checks": {"plan_exists": False},
            "go_no_go": "No-Go: frozen decoder diagnostic plan YAML not found",
        }

    with PLAN_YAML.open(encoding="utf-8") as fh:
        plan = yaml.safe_load(fh) or {}

    constraints = plan.get("constraints", {})
    constraints_ok = all(
        constraints.get(k) is True
        for k in (
            "no_train_new_decoder",
            "no_finetune_wan_vae",
            "no_continuous_video_generation",
        )
    )
    plan_steps = plan.get("plan", {})
    steps_documented = all(
        plan_steps.get(k, {}).get("description") not in (None, "")
        for k in ("step1_selection", "step2_latent_decode", "step3_report")
    )

    all_ok = constraints_ok and steps_documented
    return {
        "task_id": "M6-001",
        "experiment_id": "E-008",
        "eval_started": False,
        "checks": {
            "plan_exists": True,
            "constraints_documented": constraints_ok,
            "steps_documented": steps_documented,
        },
        "observed": {
            "constraints": constraints,
            "plan_steps": {k: v.get("status") for k, v in plan_steps.items()},
        },
        "go_no_go": (
            "TBD: frozen decoder diagnostic plan smoke passed; real decoder runs remain gated"
            if all_ok
            else "No-Go: frozen decoder diagnostic plan validation failed"
        ),
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
