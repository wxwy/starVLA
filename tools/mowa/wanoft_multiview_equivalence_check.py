"""验证双视角残差 WanOFT 在零初始化时严格保持原生动作输出。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from starVLA.model.framework.base_framework import build_framework  # noqa: E402
from starVLA.model.framework.share_tools import apply_config_compat  # noqa: E402
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils  # noqa: E402
from tools.mowa.wm4a_offline_action_check import load_episode  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/mowa/mowa_e003_wanoft_multiview_residual_stage1.yaml",
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "docs_zh/mowa/wanoft_multiview_equivalence.json",
    )
    args = parser.parse_args()

    config = apply_config_compat(OmegaConf.load(args.config))
    checkpoint = Path(config.trainer.pretrained_checkpoint)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    module_counts = {
        module: sum(key.startswith(f"{module}.") for key in state_dict)
        for module in ("backbone", "action_query_proj", "action_model")
    }
    if any(count == 0 for count in module_counts.values()):
        raise RuntimeError(f"WanOFT checkpoint module missing: {module_counts}.")
    del state_dict

    model = build_framework(config)
    TrainerUtils.load_pretrained_backbones(
        model,
        str(checkpoint),
        reload_modules=config.trainer.reload_modules,
    )
    model.to("cuda").eval()

    frames, _, instruction = load_episode(args.episode)
    example = {
        "image": [frames["primary"][args.frame], frames["wrist"][args.frame]],
        "lang": instruction,
    }

    def predict(residual_enabled: bool) -> np.ndarray:
        model.multi_view_residual_enabled = residual_enabled
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        return model.predict_action([example])["normalized_actions"]

    native = predict(False)
    residual = predict(True)
    difference = np.abs(native.astype(np.float64) - residual.astype(np.float64))
    report = {
        "checkpoint": str(checkpoint),
        "module_key_counts": module_counts,
        "episode": args.episode,
        "frame": args.frame,
        "shape": list(native.shape),
        "max_abs_diff": float(difference.max()),
        "mean_abs_diff": float(difference.mean()),
        "threshold": 1.0e-5,
        "passed": bool(difference.max() < 1.0e-5),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit("WanOFT multi-view equivalence check failed.")


if __name__ == "__main__":
    main()
