# 常见环境问题与解决方案

本文档记录在 Orion 云环境（Docker 容器 + conda Python 3.10 + CUDA 12.4 + 单 A100 80GB）下启动 StarVLA 训练时遇到的问题及其解决方法。

---

## 1. Triton: libcuda.so 找不到

**报错**：
```
AssertionError: libcuda.so cannot found!
```

**原因**：CUDA 驱动库 `libcuda.so.1` 位于非标准路径（`/opt/orion/orion_runtime/gpu/cuda/`），不在 `ldconfig` 缓存中，Triton 初始化时找不到。

**解决**：在训练脚本中添加：
```bash
export LD_LIBRARY_PATH=/opt/orion/orion_runtime/gpu/cuda:$LD_LIBRARY_PATH
```

---

## 2. Triton: Python.h 找不到

**报错**：
```
fatal error: Python.h: No such file or directory
```

**原因**：venv 从 conda 版本 Python 创建，但 `sysconfig.get_path('include')` 返回 `/usr/include/python3.10`（不存在）。Triton 编译 `driver.c` 时需要 Python 头文件，实际头文件在 conda 路径 `/root/miniconda3/envs/py310/include/python3.10/`。

**关键**：Triton 编译的是 `.c` 文件，使用 `gcc`，所以必须用 `C_INCLUDE_PATH`（`CPLUS_INCLUDE_PATH` 只对 `g++` 有效）。

**解决**：
```bash
export C_INCLUDE_PATH=/root/miniconda3/envs/py310/include/python3.10:$C_INCLUDE_PATH
```

---

## 3. GPU 数量检测为 0

**报错**：
```
assert self.local_world_size > 0
```

**原因**：`nvidia-smi -L` 在 Orion 虚拟化环境下输出为空 → `wc -l` = 0 → `num_processes=0`。

**解决**：改用 `torch.cuda.device_count()` 检测 GPU 数量：
```bash
num_processes=${NUM_PROCESSES:-$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)}
```

同理修复了以下评估脚本中依赖 `nvidia-smi` 的 GPU 检测逻辑：
- `examples/DOMINO/eval_files/start_eval.sh`
- `examples/Robotwin/eval_files/start_eval.sh`
- `examples/VLA-Arena/eval_files/run_parallel_eval.sh`

---

## 4. Qwen3.5 import 失败

**报错**：
```
ImportError: Qwen3.5 model class is unavailable. Please install transformers >= 5.2.0
```

**原因**：训练脚本配置了 `Framework_name=QwenPI` + `base_vlm=Qwen3.5-0.8B`。但 `transformers==4.57.0` 不支持 Qwen3.5（需要 >=5.2.0），且模型未下载。

**解决**：改用已有模型和兼容框架：
```bash
Framework_name=QwenGR00T    # 使用 QWen3.py，兼容 transformers 4.57
base_vlm=playground/Pretrained_models/Qwen3-VL-4B-Instruct
freeze_module_list='qwen_vl_interface'
```

> **框架兼容性参考**：
> - `QwenGR00T`、`QwenOFT`、`QwenFAST`、`QwenPI_v3` → 使用 `QWen3.py` → transformers >= 4.51
> - `QwenPI` → 使用 `QWen3_5.py` → transformers >= 5.2.0

---

## 5. NCCL bootstrap 失败

**报错**：
```
ncclInternalError: Internal check failed.
Bootstrap : no socket interface found
```

**原因**：脚本中 `export NCCL_SOCKET_IFNAME=bond0` 指向 HPC 集群的网卡绑定接口，容器中不存在。NCCL 需要找一个有效网络接口做 bootstrap。

**解决**：自动检测可用接口，按优先级选择：
```bash
if ip link show bond0 &>/dev/null; then
  export NCCL_SOCKET_IFNAME=bond0
  export NCCL_IB_HCA=mlx5_2,mlx5_3
elif ip link show eth0 &>/dev/null; then
  export NCCL_SOCKET_IFNAME=eth0
else
  export NCCL_SOCKET_IFNAME=lo
fi
```
- 单节点（任意 GPU 数）用 `lo` 即可，GPU 间通信走 PCIe/NVLink
- 多节点 HPC 自动选 `bond0` + InfiniBand
- 普通容器自动选 `eth0`

---

## 6. W&B 项目不存在

**报错**：
```
wandb.errors.errors.CommError: project not found
```

**原因**：脚本中 `wandb_entity=jinhuiye`，但实际登录用户为 `silencewx`，且项目 `starVLA_Libero` 不存在。

**解决**：
1. 在 wandb.ai 上手动创建项目 `starVLA_Libero`
2. 修改脚本 entity 为当前用户：
```bash
--wandb_entity silencewx
```

如果暂时不需要 W&B，设置 `export WANDB_MODE=disabled` 跳过。

---

## 7. 视频帧解码失败：Invalid data found when processing input

**报错**：
```
Attempt 1/10 failed for index 54981: [Errno 1094995529] Invalid data found when processing input
```

**原因**：LIBERO 数据集中某个视频文件的特定帧损坏或格式不完整，`torchvision_av` 底层 FFmpeg 无法解码。Errno `1094995529` 即 FFmpeg 的 `AVERROR_INVALIDDATA`。

**位置**：`starVLA/dataloader/gr00t_lerobot/datasets.py:2354-2355`，混合数据集 `__getitem__` 中 `get_step_data()` 读取视频帧时抛出。

**影响**：无。代码有 10 次重试机制（`max_retries=10`，line 2326），失败后随机换 index 重新采样，训练正常继续。只有连续 10 次都命中损坏帧才会真正报错退出。

**解决**：不需要处理。偶尔一两帧损坏是正常的数据噪声，不影响训练。如果频繁出现（如每几十步就报一次），检查数据集下载完整性：
```bash
# 验证 HuggingFace 下载
huggingface-cli scan-cache --repo-type dataset IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot
```

---

## 8. DataLoader worker 被 OOM Kill（checkpoint 保存时）

**报错**：
```
RuntimeError: DataLoader worker (pid 3698) is killed by signal: Killed.
```

**调用栈**：
```
train_starvla.py:333 train() → _save_checkpoint()
train_starvla.py:242 accelerator.get_state_dict(self.model)
deepspeed/checkpoint/utils.py:59 clone_tensors_for_torch_save()
```

**原因**：checkpoint 保存时 `clone_tensors_for_torch_save` 把模型参数 + DeepSpeed 优化器状态从 GPU 拷到 CPU，内存瞬间飙升。同时 `num_workers=4` 各持有数据集副本（`load_all_data_for_training: true`），系统 RAM 耗尽触发 OOM Killer。

**位置**：`starVLA/dataloader/__init__.py:48` — `num_workers=4`

**解决**：减少 DataLoader worker 数：
```python
# __init__.py line 48
num_workers=3,  # 原来是 4（或 2）
```
worker 减半 → 内存占用减半，对训练速度影响很小（I/O 不是瓶颈，GPU 是）。

> 补充：如果仍偶尔 OOM，可进一步降到 `num_workers=1` 或在脚本中限制 Python 内存：
> ```bash
> export PYTHONMALLOC=malloc
> ```

---

## 9. gradient_accumulation_steps 被 DeepSpeed 劫持，YAML 设置无效

**现象**：`starvla_cotrain_libero.yaml` 中设了 `gradient_accumulation_steps: 2`，但训练日志显示：
```
Gradient accumulation steps = 1
```

**原因**：`accelerator.gradient_accumulation_steps` 的值来源有三层，优先级混乱：

| 优先级 | 来源 | 说明 |
|--------|------|------|
| 1（最高） | `ds_config.json` | DeepSpeed ZeRO 配置文件硬编码 `gradient_accumulation_steps: 1`，当 `Accelerator(deepspeed_plugin=...)` 时直接覆盖所有其他设置 |
| 2 | YAML `trainer.gradient_accumulation_steps` | **被无视** — Accelerate+DeepSpeed 模式下不读取此值 |
| 3 | Accelerate 默认值 | 1 |

此外，单卡场景其实不需要 DeepSpeed，但原代码 `train_starvla.py:41-42` 无条件创建 `DeepSpeedPlugin()`，导致单卡也被迫走 DeepSpeed 逻辑，同时引入了上述问题。

**解决**（已修复）：

1. `train_starvla.py:41-46` — 按 GPU 数量决定是否启用 DeepSpeed：
```python
num_gpus = torch.cuda.device_count()
if num_gpus > 1:
    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config="starVLA/config/deepseeds/deepspeed_zero2.yaml")
    accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
else:
    accelerator = Accelerator(mixed_precision="bf16")
```
   - 单卡 → 无 DeepSpeed，`gradient_accumulation_steps` 默认 1，有效 batch = `per_device_batch_size`
   - 多卡 → 走 DeepSpeed，配置从 `deepspeed_zero2.yaml` → `ds_config.json` 读取

2. `run_libero_train.sh` — 去掉 `--config_file`（不再需要，Python 自行处理）：
```bash
# 改前
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${num_processes} \

# 改后
accelerate launch \
  --num_processes ${num_processes} \
  --mixed_precision bf16 \
```

> **注意**：单卡时 `gradient_accumulation_steps` 始终为 1，通过 `per_device_batch_size` 直接控制有效 batch。多卡时如需修改，改 `ds_config.json` 中的 `gradient_accumulation_steps`。

---

## 通用建议

1. **首次在新环境运行训练前**，建议先 `export WANDB_MODE=disabled` + `--is_debug True` 做 smoke test
2. **环境变量**放在训练脚本顶部，方便不同环境自行修改
3. **GPU 数量检测**优先用 `torch.cuda.device_count()`，`nvidia-smi` 在虚拟化环境下不可靠
4. **transformers 版本**升级前检查框架兼容性，QwenPI 框架需要 transformers >= 5.2.0
