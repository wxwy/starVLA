"""StarFlow-VLA 设计抽象到 StarVLA-native 实现的映射工具。"""

import json
from pathlib import Path
from typing import Any


STARFLOW_MAPPING_SCHEMA_VERSION = "p0_m3_v1"


def _get_config_value(config: Any, path: tuple[str, ...], default: Any = None) -> Any:
    value = config
    for key in path:
        if value is None:
            return default
        if isinstance(value, dict):
            value = value.get(key, default)
        else:
            value = getattr(value, key, default)
    return value


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return str(value)


def build_starflow_mapping(
    config: Any = None,
    *,
    patch_manifest_hash: str | None = None,
    starvla_commit: str | None = None,
    config_schema: str | None = None,
) -> dict[str, Any]:
    """构造可序列化的 StarFlow-VLA P0 映射记录。"""
    action_model_type = _get_config_value(
        config, ("framework", "action_model", "action_model_type"), "LayerwiseFM"
    )
    num_target_vision_tokens = _get_config_value(
        config, ("framework", "action_model", "num_target_vision_tokens"), 32
    )
    num_inference_timesteps = _get_config_value(
        config, ("framework", "action_model", "num_inference_timesteps"), 4
    )
    state_mode = _get_config_value(config, ("framework", "state_mode"), "discretized_instruction")
    version_id = _get_config_value(config, ("version_id",), None)

    return {
        "schema_version": STARFLOW_MAPPING_SCHEMA_VERSION,
        "framework_name": "StarFlowVLA",
        "implementation_mode": "starvla_native",
        "base_framework": "QwenPI_v3",
        "action_head": _to_jsonable(action_model_type),
        "state_mode": _to_jsonable(state_mode),
        "state_enters_instruction": state_mode == "discretized_instruction",
        "state_enters_action_head": state_mode == "continuous_head",
        "adapter_mode": "future_token_cross_dit",
        "flow_condition_runtime": False,
        "perceiver_enabled": False,
        "num_target_vision_tokens": _to_jsonable(num_target_vision_tokens),
        "solver": "euler",
        "num_inference_timesteps": _to_jsonable(num_inference_timesteps),
        "patch_manifest_hash": patch_manifest_hash,
        "starvla_commit": starvla_commit,
        "config_schema": config_schema or _to_jsonable(version_id),
    }


def save_starflow_mapping(path: str | Path, mapping: dict[str, Any]) -> Path:
    """保存 `starflow_mapping.json`，供 checkpoint 旁路追踪使用。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_to_jsonable(mapping), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def save_starflow_checkpoint_mapping(
    checkpoint_path: str | Path,
    config: Any = None,
    *,
    patch_manifest_hash: str | None = None,
    starvla_commit: str | None = None,
    config_schema: str | None = None,
) -> Path:
    """在 checkpoint 目录或单文件 checkpoint 旁保存 StarFlow-VLA 映射。"""
    checkpoint_path = Path(checkpoint_path)
    mapping = build_starflow_mapping(
        config,
        patch_manifest_hash=patch_manifest_hash,
        starvla_commit=starvla_commit,
        config_schema=config_schema,
    )
    if checkpoint_path.suffix:
        output_path = checkpoint_path.with_name(f"{checkpoint_path.name}.starflow_mapping.json")
    else:
        output_path = checkpoint_path / "starflow_mapping.json"
    return save_starflow_mapping(output_path, mapping)
