"""E003/E003-B2 对称双视角 Wan 辅助模块。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


def resolve_cross_view_layer_indices(
    num_blocks: int,
    *,
    num_layers: int | None = 10,
    start_layer_ratio: float = 2.0 / 3.0,
) -> tuple[int, ...]:
    """解析参与 cross-view attention 的尾部 Wan block 下标。"""

    if num_layers is None:
        start_layer = max(0, min(num_blocks, int(num_blocks * start_layer_ratio)))
    else:
        if num_layers <= 0 or num_layers > num_blocks:
            raise ValueError(f"num_layers must be in [1, {num_blocks}], got {num_layers}.")
        start_layer = num_blocks - num_layers
    return tuple(range(start_layer, num_blocks))


def flatten_view_batch(video_latents: torch.Tensor) -> torch.Tensor:
    """按 sample-major/view-minor 顺序将 [B,V,C,T,H,W] 展开为 [B*V,C,T,H,W]。"""

    if video_latents.dim() != 6:
        raise ValueError(f"video_latents must be [B,V,C,T,H,W], got {tuple(video_latents.shape)}.")
    return video_latents.reshape(
        video_latents.shape[0] * video_latents.shape[1], *video_latents.shape[2:]
    )


def unflatten_view_batch(hidden: torch.Tensor, batch_size: int, num_views: int) -> torch.Tensor:
    """恢复显式视角维，保持 main_i/wrist_i 配对。"""

    if hidden.shape[0] != batch_size * num_views:
        raise ValueError(
            f"Flattened batch mismatch: got {hidden.shape[0]}, expected {batch_size}*{num_views}."
        )
    return hidden.reshape(batch_size, num_views, *hidden.shape[1:])


@dataclass(frozen=True)
class MultiViewPatchGrid:
    batch_size: int
    num_views: int
    time: int
    height: int
    width: int

    @property
    def spatial_tokens(self) -> int:
        return self.height * self.width

    @property
    def sequence_tokens(self) -> int:
        return self.time * self.spatial_tokens


class CrossViewAttentionAdapter(nn.Module):
    """只在相同 latent timestep 内交换不同视角的空间 token。"""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        *,
        bottleneck_dim: int | None = None,
        gate_init: float = 0.0,
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()
        bottleneck_dim = hidden_dim if bottleneck_dim is None else int(bottleneck_dim)
        if bottleneck_dim <= 0 or bottleneck_dim % num_heads != 0:
            raise ValueError(
                f"bottleneck_dim must be positive and divisible by num_heads, got "
                f"{bottleneck_dim} and {num_heads}."
            )
        self.norm = nn.LayerNorm(hidden_dim)
        self.input_projection = nn.Linear(hidden_dim, bottleneck_dim)
        self.attention = nn.MultiheadAttention(bottleneck_dim, num_heads, batch_first=True)
        self.output_projection = nn.Linear(bottleneck_dim, hidden_dim)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.register_buffer("last_output_norm", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_grad_norm", torch.tensor(0.0), persistent=False)
        if zero_init_output:
            # 极小初始化兼顾近似恒等起点与首个 backward 的可观测梯度。
            nn.init.normal_(self.output_projection.weight, std=1e-5)
            nn.init.zeros_(self.output_projection.bias)

    def forward(self, hidden: torch.Tensor, grid: MultiViewPatchGrid) -> torch.Tensor:
        if hidden.shape[1] != grid.sequence_tokens:
            raise ValueError(
                f"Wan token/grid mismatch: tokens={hidden.shape[1]}, grid={grid.sequence_tokens}."
            )
        paired = unflatten_view_batch(hidden, grid.batch_size, grid.num_views)
        paired = paired.reshape(
            grid.batch_size,
            grid.num_views,
            grid.time,
            grid.spatial_tokens,
            hidden.shape[-1],
        )
        aligned = paired.permute(0, 2, 1, 3, 4).reshape(
            grid.batch_size * grid.time,
            grid.num_views * grid.spatial_tokens,
            hidden.shape[-1],
        )
        normalized = self.input_projection(self.norm(aligned))
        delta, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        delta = self.output_projection(delta)
        self.last_output_norm.copy_(delta.detach().float().norm())
        if delta.requires_grad:
            def _record_grad_norm(grad: torch.Tensor) -> torch.Tensor:
                self.last_grad_norm.copy_(grad.detach().float().norm())
                return grad

            delta.register_hook(_record_grad_norm)
        restored = delta.reshape(
            grid.batch_size,
            grid.time,
            grid.num_views,
            grid.spatial_tokens,
            hidden.shape[-1],
        ).permute(0, 2, 1, 3, 4)
        restored = restored.reshape_as(paired).reshape_as(unflatten_view_batch(hidden, grid.batch_size, grid.num_views))
        return hidden + self.gate.to(dtype=hidden.dtype) * flatten_view_batch_hidden(restored)


def flatten_view_batch_hidden(hidden: torch.Tensor) -> torch.Tensor:
    """将 [B,V,N,D] 展开为 [B*V,N,D]。"""

    if hidden.dim() != 4:
        raise ValueError(f"hidden must be [B,V,N,D], got {tuple(hidden.shape)}.")
    return hidden.reshape(hidden.shape[0] * hidden.shape[1], hidden.shape[2], hidden.shape[3])


class MultiViewFutureFusion(nn.Module):
    """用少量可学习 query 将双视角 future hidden 适配回 LayerwiseFM 条件。"""

    def __init__(self, hidden_dim: int, num_heads: int, num_queries: int) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim) / math.sqrt(hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

    def forward(self, future_hidden: torch.Tensor) -> torch.Tensor:
        if future_hidden.dim() != 4:
            raise ValueError(
                f"future_hidden must be [B,V,N_future,D], got {tuple(future_hidden.shape)}."
            )
        batch_size = future_hidden.shape[0]
        key_value = future_hidden.reshape(batch_size, -1, future_hidden.shape[-1])
        key_value = self.norm(key_value.to(dtype=self.norm.weight.dtype)).to(dtype=future_hidden.dtype)
        queries = self.queries.expand(batch_size, -1, -1).to(dtype=future_hidden.dtype)
        fused, _ = self.attention(queries, key_value, key_value, need_weights=False)
        return fused


class MultiViewDoneHead(nn.Module):
    """从同一未来 timestep 的主/腕视角 Wan hidden 预测共享终止概率。"""

    def __init__(self, hidden_dim: int, mlp_hidden_dim: int = 512) -> None:
        super().__init__()
        self.view_embeddings = nn.Embedding(2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, 1),
        )
        nn.init.normal_(self.view_embeddings.weight, std=1e-3)

    def forward(self, future_hidden: torch.Tensor) -> torch.Tensor:
        """Args: future_hidden [B,V=2,T_future,S_patch,D]."""

        if future_hidden.dim() != 5 or future_hidden.shape[1] != 2:
            raise ValueError(
                "future_hidden must be [B,V=2,T_future,S_patch,D], "
                f"got {tuple(future_hidden.shape)}."
            )
        spatial_pooled = future_hidden.mean(dim=3)
        view_ids = torch.arange(2, device=future_hidden.device)
        view_features = spatial_pooled + self.view_embeddings(view_ids)[None, :, None, :].to(
            dtype=future_hidden.dtype
        )
        fused = torch.cat((view_features[:, 0], view_features[:, 1]), dim=-1)
        fused = self.norm(fused.to(dtype=self.norm.weight.dtype))
        return self.mlp(fused).squeeze(-1)


def masked_future_flow_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """对每个视角按自身有效 future latent 数归一化。"""

    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target mismatch: {prediction.shape} != {target.shape}.")
    if prediction.dim() != 6:
        raise ValueError("prediction must be [B,V,C,T,H,W].")
    if valid_mask.shape != prediction.shape[:2] + (prediction.shape[3],):
        raise ValueError(
            f"valid_mask must be [B,V,T], got {tuple(valid_mask.shape)} for {tuple(prediction.shape)}."
        )
    token_loss = (prediction.float() - target.float()).pow(2).mean(dim=(2, 4, 5))
    weights = valid_mask.to(dtype=token_loss.dtype)
    return (token_loss * weights).sum(dim=-1) / weights.sum(dim=-1).clamp(min=1)
