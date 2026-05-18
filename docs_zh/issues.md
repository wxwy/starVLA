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

## 通用建议

1. **首次在新环境运行训练前**，建议先 `export WANDB_MODE=disabled` + `--is_debug True` 做 smoke test
2. **环境变量**放在训练脚本顶部，方便不同环境自行修改
3. **GPU 数量检测**优先用 `torch.cuda.device_count()`，`nvidia-smi` 在虚拟化环境下不可靠
4. **transformers 版本**升级前检查框架兼容性，QwenPI 框架需要 transformers >= 5.2.0
