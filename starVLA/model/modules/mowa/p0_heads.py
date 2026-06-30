"""MoWA P0 ConstructibleHeads smoke module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


MOWA_P0_CONSTRUCTIBLE_HEADS = ("task_progress", "action_outcome_class")


@dataclass(frozen=True)
class MoWAP0ConstructibleHeadsConfig:
    input_dim: int = 4
    hidden_dim: int = 32


def mowa_manual_sgd_step(model: Any, lr: float) -> None:
    """Apply one tiny SGD step without importing torch optimizer stacks."""

    import torch

    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter -= lr * parameter.grad
                parameter.grad = None


class MoWAP0ConstructibleHeads:
    """Tiny P0 head module for train-smoke only.

    该模块只覆盖当前 G0 已放行的两个 ConstructibleHeads：
    `task_progress` 和 `action_outcome_class`。其余 P0 head 不在本模块中
    偷偷启用，必须继续由 mask 处理。
    """

    def __new__(cls, *args: Any, **kwargs: Any):
        import torch.nn as nn

        class _TorchMoWAP0ConstructibleHeads(nn.Module):
            def __init__(self, config: MoWAP0ConstructibleHeadsConfig | None = None):
                super().__init__()
                self.config = config or MoWAP0ConstructibleHeadsConfig()
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
                    loss = F.mse_loss(
                        outputs["action_outcome_class"],
                        targets["action_outcome_class"],
                    )
                    losses["action_outcome_class"] = loss
                    active_losses.append(loss)
                if not active_losses:
                    raise ValueError("MoWA P0 ConstructibleHeads loss requires at least one active mask.")
                total = torch.stack(active_losses).sum()
                losses["total"] = total
                return total, losses, outputs

        return _TorchMoWAP0ConstructibleHeads(*args, **kwargs)


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
            "class_mapping_status": "Data Gate",
            "source": "G0 smoke labels",
        },
    }
