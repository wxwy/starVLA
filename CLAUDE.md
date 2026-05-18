# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

StarVLA 是一个模块化的 Vision-Language-Action (VLA) 模型开发平台，设计理念为"乐高式"即插即用。每个功能组件（模型、数据、训练器、配置、评估）遵循高内聚低耦合原则。

## 常用命令

```bash
# 代码风格检查与格式化
make check          # black + ruff 检查（不修改文件）
make autoformat     # 自动格式化
make clean          # 清理 pyc/__pycache__

# 单独运行模块进行 smoke test
python starVLA/model/framework/VLM4A/QwenOFT.py --config_yaml starvla_cotrain_oxe.yaml
python starVLA/dataloader/lerobot_datasets.py --config_yaml starvla_cotrain_oxe.yaml

# 典型训练启动（LIBERO 为例）
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/LIBERO/train_files/starvla_cotrain_libero.yaml \
  --framework.name QwenOFT \
  --run_root_dir ./output \
  --run_id my_experiment
```

## 核心架构

### 框架注册机制
所有 VLA 框架通过 `@FRAMEWORK_REGISTRY.register("框架名")` 自动注册。`build_framework(cfg)` 根据 `cfg.framework.name` 查找并实例化。新增框架只需在 `starVLA/model/framework/VLM4A/` 或 `WM4A/` 下添加文件即可自动发现。

### 目录职责
- **`starVLA/model/framework/`** — VLA 框架定义（唯一对外 API 入口）。`VLM4A/` 为 VLM→Action 框架，`WM4A/` 为 WorldModel→Action 框架。每个框架文件可独立运行做 smoke test。关键变体：QwenOFT（并行连续动作）、QwenFast（自回归离散动作）、QwenPI（流匹配扩散动作）、QwenGR00T（双系统架构）
- **`starVLA/model/modules/`** — 可复用子模块：`action_model/`（动作头 FAST/OFT/PI/GR00T/DiT）、`vlm/`（Qwen2.5/3/3.5、Florence2、Gemma4、Molmo2）、`world_model/`（CosmosPredict2、Wan2.2）、`projector/`（QFormer）、`dino_model/`
- **`starVLA/dataloader/`** — 数据加载，返回原始 model-agnostic dict（image/list[PIL], lang/str, action/np.ndarray, state/optional）。基于 LeRobot 格式，`gr00t_lerobot/` 为底层实现
- **`starVLA/training/`** — 三个训练脚本：`train_starvla.py`（VLA训练）、`train_starvlm.py`（VLM训练）、`train_starvla_cotrain.py`（VLA+VLM联合训练）。基于原生 PyTorch + Accelerate + DeepSpeed
- **`starVLA/config/`** — `deepseeds/` 为 DeepSpeed 配置（zero2/zero3），`training/` 为训练 YAML 配置
- **`examples/`** — 各 benchmark 示例（LIBERO、SimplerEnv、RoboCasa、RoboTwin、DOMINO、BEHAVIOR、Calvin 等）
- **`deployment/`** — 模型部署：`model_server/` 推理服务，`upload/` 模型上传

### 配置系统
- 使用 OmegaConf YAML 配置 + CLI 覆盖（`--xxx.yyy=value` 格式）
- `AccessTrackedConfig` 包裹配置，仅保存实际被访问的配置项
- 支持按模块名设置不同学习率（`trainer.learning_rate` 字典）和冻结模块（`trainer.freeze_modules` 逗号分隔列表）

### 数据流
Dataloader → 原始 dict → `framework.forward()` / `framework.predict_action()` → 动作输出。数据与模型完全解耦，dataloader 不做 tokenizer/图像编码等模型特定预处理。

## 注意事项
- `**/bar/` 目录被 gitignore，可在其中放置自定义脚本
- 框架名通过 `@FRAMEWORK_REGISTRY.register("name")` 注册，`--framework.name` 选择
- `--framework.qwenvl.base_vlm` 参数名历史遗留，实际支持多种 VLM
- 脚本中硬编码了 `Qwen/Qwen2.5-VL-7B-Instruct` 等默认 VLM 路径
