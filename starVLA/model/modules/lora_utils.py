"""LoRA 注入模型的预训练权重兼容加载工具。"""

from __future__ import annotations

from collections.abc import Mapping

import torch


def load_pretrained_lora_compatible(
    module: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    optional_prefixes: tuple[str, ...] = (),
) -> torch.nn.modules.module._IncompatibleKeys:
    """加载 LoRA 注入前导出的权重，并严格拒绝真实骨干缺失。"""
    current_keys = set(module.state_dict().keys())
    remapped: dict[str, torch.Tensor] = {}
    unmatched: list[str] = []

    for key, value in state_dict.items():
        candidates = [key]
        if key.startswith("model."):
            candidates.append("model.base_model.model." + key[len("model.") :])
        for candidate in list(candidates):
            for suffix in (".weight", ".bias"):
                if candidate.endswith(suffix):
                    candidates.append(
                        candidate[: -len(suffix)] + ".base_layer" + suffix
                    )
        target = next((candidate for candidate in candidates if candidate in current_keys), None)
        if target is None:
            if not key.startswith(optional_prefixes):
                unmatched.append(key)
            continue
        remapped[target] = value

    result = module.load_state_dict(remapped, strict=False)
    missing_non_lora = [key for key in result.missing_keys if "lora_" not in key]
    if unmatched or result.unexpected_keys or missing_non_lora:
        raise RuntimeError(
            "LoRA-compatible checkpoint mismatch: "
            f"unmatched={unmatched[:8]}, unexpected={result.unexpected_keys[:8]}, "
            f"missing={missing_non_lora[:8]}"
        )
    return result
