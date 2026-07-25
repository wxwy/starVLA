"""固定 LIBERO batch 上验证 WanOFT 多视角 fusion 的动作过拟合能力。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from starVLA.model.framework.base_framework import build_framework  # noqa: E402
from starVLA.model.framework.share_tools import apply_config_compat  # noqa: E402
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils  # noqa: E402
from tools.mowa.wm4a_offline_action_check import load_episode  # noqa: E402


def normalize_actions(actions: np.ndarray, statistics: dict) -> torch.Tensor:
    """复用 LIBERO 原生 data config：前6维 min_max，夹爪保持二值 0/1。"""
    normalized = actions.astype(np.float32).copy()
    minimum = np.asarray(statistics["min"][:6], dtype=np.float32)
    maximum = np.asarray(statistics["max"][:6], dtype=np.float32)
    normalized[..., :6] = 2.0 * (normalized[..., :6] - minimum) / (maximum - minimum) - 1.0
    normalized[..., :6] = np.clip(normalized[..., :6], -1.0, 1.0)
    normalized[..., 6] = (normalized[..., 6] > 0.5).astype(np.float32)
    return torch.from_numpy(normalized)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=REPO / "configs/mowa/mowa_e003_wanoft_multiview_residual_stage1.yaml",
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        default=REPO / "playground/mowa_ckpt/MoWA-E-003-WanOFT-multiview-residual-overfit/checkpoints/steps_1000",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "docs_zh/mowa/wanoft_multiview_fusion_overfit.json",
    )
    args = parser.parse_args()
    if args.samples <= 0 or args.steps <= 0:
        raise ValueError("--samples and --steps must be positive.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = apply_config_compat(OmegaConf.load(args.config))
    model = build_framework(config)
    TrainerUtils.load_pretrained_backbones(
        model, str(config.trainer.pretrained_checkpoint), reload_modules=config.trainer.reload_modules,
    )
    model.to("cuda").eval()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("multi_view_fusion."))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if len(trainable) != 2:
        raise RuntimeError(f"Expected fusion weight/bias only, got {len(trainable)} trainable tensors.")
    initial_fusion = [parameter.detach().clone() for parameter in trainable]

    frames, actions, instruction = load_episode(args.episode)
    if args.samples + model.action_horizon > len(actions):
        raise ValueError("Requested fixed batch exceeds episode action length.")
    stats = json.loads((Path(config.trainer.pretrained_checkpoint).parents[1] / "dataset_statistics.json").read_text())
    targets = normalize_actions(
        np.stack([actions[index : index + model.action_horizon] for index in range(args.samples)]),
        stats["franka"]["action"],
    ).cuda()

    base_features, wrist_features = [], []
    with torch.no_grad():
        for index in range(args.samples):
            base, wrist = model._extract_base_and_wrist_features(
                [[frames["primary"][index], frames["wrist"][index]]], [instruction]
            )
            base_features.append(base.float().cpu())
            wrist_features.append(wrist.float().cpu())
    base_features = torch.cat(base_features).cuda()
    wrist_features = torch.cat(wrist_features).cuda()

    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
    with torch.no_grad():
        initial_prediction = model.action_model.predict_action(
            model._pool_to_action_queries(model._fuse_action_features(base_features, wrist_features))
        )
        initial_l1 = float(F.l1_loss(initial_prediction, targets))
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        fused = model._fuse_action_features(base_features, wrist_features)
        prediction = model.action_model.predict_action(model._pool_to_action_queries(fused))
        loss = F.l1_loss(prediction, targets)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_prediction = model.action_model.predict_action(
            model._pool_to_action_queries(model._fuse_action_features(base_features, wrist_features))
        )
        final_l1 = float(F.l1_loss(final_prediction, targets))
        gripper_accuracy = float(((final_prediction[..., 6] > 0.5) == (targets[..., 6] > 0.5)).float().mean())
        fusion_delta = float(
            sum(
                (parameter.detach().float() - initial.detach().float()).square().sum()
                for parameter, initial in zip(trainable, initial_fusion)
            ).sqrt()
        )

    consistency_example = {
        "image": [frames["primary"][0], frames["wrist"][0]],
        "lang": instruction,
        "action": targets[0].cpu().numpy(),
    }
    consistency_seed = args.seed + 1000
    torch.manual_seed(consistency_seed)
    torch.cuda.manual_seed_all(consistency_seed)
    with torch.no_grad():
        direct_prediction = model.action_model.predict_action(
            model._pool_to_action_queries(model._extract_action_feature(
                [consistency_example["image"]], [instruction]
            ))
        )
    torch.manual_seed(consistency_seed)
    torch.cuda.manual_seed_all(consistency_seed)
    predict_prediction = torch.from_numpy(
        model.predict_action([consistency_example])["normalized_actions"]
    ).to(device=direct_prediction.device, dtype=direct_prediction.dtype)
    torch.manual_seed(consistency_seed)
    torch.cuda.manual_seed_all(consistency_seed)
    train_loss = model([consistency_example])["action_loss"]
    direct_loss = F.l1_loss(direct_prediction, targets[:1])
    predict_max_abs_diff = float((direct_prediction - predict_prediction).abs().max())
    train_loss_abs_diff = float((train_loss - direct_loss).abs())
    report = {
        "episode": args.episode,
        "samples": args.samples,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "initial_l1": initial_l1,
        "final_l1": final_l1,
        "gripper_accuracy": gripper_accuracy,
        "fusion_parameter_l2": fusion_delta,
        "predict_max_abs_diff": predict_max_abs_diff,
        "train_loss_abs_diff": train_loss_abs_diff,
        "passed": bool(
            final_l1 < 0.02
            and gripper_accuracy > 0.99
            and predict_max_abs_diff < 1.0e-5
            and train_loss_abs_diff < 1.0e-5
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file

    lightweight_state = {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in model.state_dict().items()
        if not name.startswith("backbone.")
    }
    save_file(lightweight_state, str(args.checkpoint_dir / "model.safetensors"))
    (args.checkpoint_dir / "trainer_state.json").write_text(
        json.dumps({"omitted_model_state_prefixes": ["backbone."]}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit("WanOFT fusion fixed-batch overfit did not meet acceptance thresholds.")


if __name__ == "__main__":
    main()
