"""MoWA Future Feature Heads module (formerly P0 Heads)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from starVLA.mowa_constants import (
    MOWA_ACTION_OUTCOME_CLASS_MAPPING_NOTE,
    MOWA_ACTION_OUTCOME_CLASS_MAPPING_STATUS,
    MOWA_ACTION_OUTCOME_CLASS_MAPPING_VERSION,
    MOWA_FUTURE_FULL_HEADS,
    MOWA_FUTURE_HEAD_OUTPUT_DIMS,
)


@dataclass(frozen=True)
class MoWAFutureConstructibleHeadsConfig:
    input_dim: int = 4
    hidden_dim: int = 32
    action_outcome_loss_type: str = "mse"


@dataclass(frozen=True)
class MoWAFutureFullHeadsConfig:
    input_dim: int = 4
    hidden_dim: int = 32
    action_outcome_loss_type: str = "mse"


@dataclass(frozen=True)
class MoWAFutureFeatures:
    hidden_features: Any
    head_outputs: Mapping[str, Any]
    active_heads: tuple[str, ...]
    masked_heads: tuple[str, ...]


def mowa_manual_sgd_step(model: Any, lr: float) -> None:
    """Apply one tiny SGD step without importing torch optimizer stacks."""

    import torch

    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter -= lr * parameter.grad
                parameter.grad = None


class MoWAFutureConstructibleHeads:
    """Constructible future-feature head module for train-smoke only.

    该模块只覆盖当前 G0 已放行的两个 constructible heads：
    `task_progress` 和 `action_outcome_class`。其余 head 不在本模块中
    偷偷启用，必须继续由 mask 处理。
    """

    def __new__(cls, *args: Any, **kwargs: Any):
        import torch.nn as nn

        class _TorchMoWAFutureConstructibleHeads(nn.Module):
            def __init__(self, config: MoWAFutureConstructibleHeadsConfig | None = None):
                super().__init__()
                self.config = config or MoWAFutureConstructibleHeadsConfig()
                self.trunk = nn.Sequential(
                    nn.Linear(self.config.input_dim, self.config.hidden_dim),
                    nn.ReLU(),
                )
                self.task_progress = nn.Linear(self.config.hidden_dim, 1)
                self.action_outcome_class = nn.Linear(self.config.hidden_dim, 2)

            def forward(self, features):
                hidden = self.trunk(features)
                return {
                    "task_progress": self.task_progress(hidden).squeeze(-1),
                    "action_outcome_class": self.action_outcome_class(hidden),
                }

            def compute_loss(self, features, targets: Mapping[str, Any], masks: Mapping[str, Any]):
                import torch
                import torch.nn.functional as F

                outputs = self(features)
                losses = {}
                active_losses = []
                if bool(masks.get("task_progress", False)):
                    loss = F.mse_loss(outputs["task_progress"], targets["task_progress"])
                    losses["task_progress"] = loss
                    active_losses.append(loss)
                if bool(masks.get("action_outcome_class", False)):
                    loss = _compute_mowa_head_loss(
                        "action_outcome_class",
                        outputs["action_outcome_class"],
                        targets["action_outcome_class"],
                        action_outcome_loss_type=self.config.action_outcome_loss_type,
                    )
                    losses["action_outcome_class"] = loss
                    active_losses.append(loss)
                if not active_losses:
                    raise ValueError("MoWA P0 ConstructibleHeads loss requires at least one active mask.")
                total = torch.stack(active_losses).sum()
                losses["total"] = total
                return total, losses, outputs

        return _TorchMoWAFutureConstructibleHeads(*args, **kwargs)


class MoWAFutureFullHeads:
    """Seven-head future-feature interface with mask-controlled loss.

    该模块固定七类 future heads；缺失标签必须通过 mask 关闭，不新增替代 head。
    """

    def __new__(cls, *args: Any, **kwargs: Any):
        import torch.nn as nn

        class _TorchMoWAFutureFullHeads(nn.Module):
            def __init__(self, config: MoWAFutureFullHeadsConfig | None = None):
                super().__init__()
                self.config = config or MoWAFutureFullHeadsConfig()
                self.trunk = nn.Sequential(
                    nn.Linear(self.config.input_dim, self.config.hidden_dim),
                    nn.ReLU(),
                )
                self.heads = nn.ModuleDict(
                    {
                        head: nn.Linear(self.config.hidden_dim, output_dim)
                        for head, output_dim in MOWA_FUTURE_HEAD_OUTPUT_DIMS.items()
                    }
                )

            def forward(self, features):
                hidden = self.trunk(features)
                outputs = {}
                for head, module in self.heads.items():
                    value = module(hidden)
                    outputs[head] = value.squeeze(-1) if value.shape[-1] == 1 else value
                return outputs

            def future_features(self, features, masks: Mapping[str, Any]) -> MoWAFutureFeatures:
                hidden = self.trunk(features)
                outputs = {}
                for head, module in self.heads.items():
                    value = module(hidden)
                    outputs[head] = value.squeeze(-1) if value.shape[-1] == 1 else value
                active_heads = tuple(head for head in MOWA_FUTURE_FULL_HEADS if bool(masks.get(head, False)))
                masked_heads = tuple(head for head in MOWA_FUTURE_FULL_HEADS if not bool(masks.get(head, False)))
                return MoWAFutureFeatures(
                    hidden_features=hidden,
                    head_outputs=outputs,
                    active_heads=active_heads,
                    masked_heads=masked_heads,
                )

            def compute_loss(self, features, targets: Mapping[str, Any], masks: Mapping[str, Any]):
                import torch

                outputs = self(features)
                losses = {}
                active_losses = []
                for head in MOWA_FUTURE_FULL_HEADS:
                    if not bool(masks.get(head, False)):
                        continue
                    if head not in targets:
                        raise KeyError(f"MoWA P0 FullHeads active head missing target: {head}")
                    loss = _compute_mowa_head_loss(
                        head,
                        outputs[head],
                        targets[head],
                        action_outcome_loss_type=self.config.action_outcome_loss_type,
                    )
                    losses[head] = loss
                    active_losses.append(loss)
                if not active_losses:
                    raise ValueError("MoWA P0 FullHeads loss requires at least one active mask.")
                total = torch.stack(active_losses).sum()
                losses["total"] = total
                return total, losses, outputs

        return _TorchMoWAFutureFullHeads(*args, **kwargs)


def _compute_mowa_head_loss(
    head: str,
    output: Any,
    target: Any,
    *,
    action_outcome_loss_type: str,
) -> Any:
    import torch
    import torch.nn.functional as F

    if head != "action_outcome_class":
        return F.mse_loss(output, target)
    if action_outcome_loss_type == "mse":
        return F.mse_loss(output, target)
    if action_outcome_loss_type == "cross_entropy_done":
        target_tensor = torch.as_tensor(target, device=output.device)
        if target_tensor.ndim > 1:
            target_tensor = target_tensor[..., -1]
        return F.cross_entropy(output.float(), target_tensor.long())
    raise ValueError(f"Unsupported MoWA action_outcome_loss_type: {action_outcome_loss_type}")


# Backward-compatible P0 aliases (deprecated — prefer Future* names).
MoWAP0ConstructibleHeadsConfig = MoWAFutureConstructibleHeadsConfig
MoWAP0FullHeadsConfig = MoWAFutureFullHeadsConfig
P0FutureFeatures = MoWAFutureFeatures
MoWAP0ConstructibleHeads = MoWAFutureConstructibleHeads
MoWAP0FullHeads = MoWAFutureFullHeads
MoWAFutureFeatureHeadsConfig = MoWAFutureFullHeadsConfig  # convenient alias
MoWAFutureFeatureHeads = MoWAFutureFullHeads             # convenient alias


def build_mowa_p0_constructible_batch_from_smoke(
    data_root: Path | str,
    *,
    max_samples: int = 32,
) -> dict[str, Any]:
    """Build a small tensor batch from G0 smoke labels.

    This is a train-smoke bridge, not a production dataloader.
    """

    import torch

    from starVLA.dataloader.mowa.p0_label_builder import build_mowa_p0_label_smoke_sample
    from starVLA.dataloader.mowa.robocasa365_recipe import (
        MOWA_ROBOCASA365_TARGET_HUMAN_ATOMIC_CORE_TASK_PATHS,
    )

    root = Path(data_root)
    features = []
    progress_targets = []
    outcome_targets = []
    sample_meta = []
    for task, relative_path in MOWA_ROBOCASA365_TARGET_HUMAN_ATOMIC_CORE_TASK_PATHS.items():
        dataset_path = root / relative_path
        for episode_index in (0, 1, 4):
            label = build_mowa_p0_label_smoke_sample(
                dataset_path,
                episode_index=episode_index,
                row_index=3,
            )
            progress = float(label.labels["task_progress"])
            outcome = label.labels["action_outcome_class"]
            reward = float(outcome["next_reward"])
            done = 1.0 if bool(outcome["next_done"]) else 0.0
            features.append([progress, reward, done, 1.0])
            progress_targets.append(progress)
            outcome_targets.append([reward, done])
            sample_meta.append(
                {
                    "task": task,
                    "relative_path": relative_path,
                    "episode_index": episode_index,
                    "row_index": 3,
                }
            )
            if len(features) >= max_samples:
                break
        if len(features) >= max_samples:
            break

    return {
        "features": torch.tensor(features, dtype=torch.float32),
        "targets": {
            "task_progress": torch.tensor(progress_targets, dtype=torch.float32),
            "action_outcome_class": torch.tensor(outcome_targets, dtype=torch.float32),
        },
        "masks": {
            "task_progress": True,
            "action_outcome_class": True,
        },
        "metadata": {
            "sample_count": len(features),
            "sample_meta": sample_meta,
            "class_mapping_status": MOWA_ACTION_OUTCOME_CLASS_MAPPING_STATUS,
            "class_mapping_version": MOWA_ACTION_OUTCOME_CLASS_MAPPING_VERSION,
            "class_mapping_note": MOWA_ACTION_OUTCOME_CLASS_MAPPING_NOTE,
            "source": "G0 smoke labels",
        },
    }


build_mowa_future_constructible_batch_from_smoke = build_mowa_p0_constructible_batch_from_smoke
