# Quick Start: Train & Evaluate Your First VLA with LIBERO

This guide walks you through the complete StarVLA workflow — from installation to training to evaluation — using the **LIBERO** benchmark as a concrete, end-to-end example. By the end you will have a trained VLA policy and know how to evaluate it in simulation.

> **Just want to evaluate a released checkpoint?** Jump to [Evaluate a Pretrained Checkpoint](#-evaluate-a-pretrained-checkpoint) — no training required.
>
> **Bringing your own dataset / robot?** Read [`integrate_your_dataset.md`](integrate_your_dataset.md)
> for the end-to-end "use my own data" guide, or activate the bundled
> [agent skill](agent_skills/integrate-starvla-dataset/README.md) to have a
> code agent (Claude Code, VS Code Copilot, …) drive the integration for you.

---

## Table of Contents

- [0. Installation](#0-installation)
- [1. Verify Your Installation](#1-verify-your-installation)
- [2. Prepare Training Data](#2-prepare-training-data)
- [3. Prepare Pretrained Models](#3-prepare-pretrained-models)
- [4. Understanding the Training Config](#4-understanding-the-training-config)
- [5. Training Data Flow (conceptual)](dataflow.md)
- [5. Understanding the Training Script](#5-understanding-the-training-script)
- [6. Launch Training](#6-launch-training)
- [7. Evaluate a Pretrained Checkpoint](#7-evaluate-a-pretrained-checkpoint)
- [Next Steps](#next-steps)

---

## 0. Installation

```bash
git clone https://github.com/starVLA/starVLA
cd starVLA

conda create -n starVLA python=3.10 -y
conda activate starVLA

pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install -e .
```

> **wxwy note: flash-attn version pinned to 2.7.4.post1.**
>
> Rationale: Current environment is Python 3.10 + torch 2.6.0+cu124 + CUDA 12.4. Compared to source-compiling flash-attn 2.8.x, version 2.7.4.post1 can be installed directly with official pre-built wheels, avoiding long CUDA compilation time, OOM, GitHub download interruptions, etc. — more suitable for stable cloud server deployment of starVLA.
>
> Alternatively, download the pre-built wheel directly:
> ```bash
> wget -c "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
> pip install ./flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
> ```

<details>
<summary><b>⚠️ Common Issues</b></summary>

flash-attn can be tricky to install because it must match your system's CUDA toolkit (nvcc) and PyTorch versions. The `--no-build-isolation` flag resolves most issues, but on newer systems you may need to manually choose a compatible flash-attn version. Ensure your CUDA driver/toolkit and torch versions are aligned. Check your environment:

```bash
nvcc -V
pip list | grep -E 'torch|transformers|flash-attn'
```

If issues persist, pick a flash-attn release that matches your versions (CUDA and torch) or ask ChatGPT with its search function for help with the outputs above.

We have verified that `flash-attn==2.7.4.post1` works well with nvcc versions `12.0` and `12.4`.

</details>

<details>
<summary><b>⚠️ flash-attn 2.7.4.post1 limitations vs 2.8.x (wxwy note)</b></summary>

1. Does not include new or enhanced Hopper / Blackwell / FP8 / cute kernel support introduced in 2.8.x.
2. Performance optimizations for newer architectures (H100, B100/B200, sm90/sm100) are inferior to 2.8.x.
3. If future project work explicitly requires flash-attn 2.8.x new APIs or new kernels, re-evaluate upgrading.
4. Minimal impact on current sm86/Ampere GPUs and starVLA training/inference scenarios — stability is prioritized over new features.

</details>

---

## 1. Verify Your Installation

Run a quick smoke test to make sure the framework loads correctly:

```bash
python starVLA/model/framework/VLM4A/QwenGR00T.py
```

This requires [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) at `./playground/Pretrained_models/Qwen3-VL-4B-Instruct` (see [Step 3](#3-prepare-pretrained-models)). It should print the model architecture and run a forward pass on fake data without errors.

### Verify Flash Attention

The standalone `flash-attn` package is critical for training throughput. Run the bundled test to check import, CUDA kernel, and performance vs. PyTorch's built-in SDPA:

```bash
python docs_zh/test_flash_attn.py
```

This script checks three things:
1. Whether `flash_attn` can be imported
2. Whether `flash_attn_func` CUDA kernel runs correctly
3. Performance benchmark: `flash_attn_func` vs. `F.scaled_dot_product_attention`

**Expected sample output** (GPU: A100 / sm80, torch 2.6.0+cu124, flash_attn 2.7.4.post1):

```
============================================================
PyTorch 信息
============================================================
torch version: 2.6.0+cu124
cuda available: True
cuda device: P1.gpu.medium
compute capability: (8, 0)

============================================================
FlashAttention 导入检测
============================================================
flash_attn import: SUCCESS
flash_attn version: 2.7.4.post1

============================================================
FlashAttention CUDA Kernel 检测
============================================================
flash_attn_func: SUCCESS
output shape: torch.Size([2, 128, 8, 64])
output dtype: torch.float16

============================================================
FlashAttention 性能对比
============================================================
SDPA backends: {'flash_sdp': True, 'mem_efficient': True, 'math': True}
PyTorch SDPA  : 0.0155 sec (avg 0.309 ms/iter)
flash_attn    : 0.0096 sec (avg 0.192 ms/iter)
加速比        : 1.61x (flash_attn 更快)

============================================================
FlashAttention 安装正常
============================================================
```

The key output is the **speedup ratio** (> 1.0x means flash_attn is faster than PyTorch's default SDPA path). If the ratio is ~1.0x, your PyTorch is likely already dispatching to its own flash-sdp backend under the hood.

---

## 2. Prepare Training Data

StarVLA uses **LeRobot-format** datasets. We provide a one-command script to download all four LIBERO suites (Spatial, Object, Goal, Long-Horizon) plus the co-training VLM data:

```bash
# Set DEST to where you want to store the raw data (can be a shared disk)
export DEST=/path/to/your/data/directory
bash examples/LIBERO/data_preparation.sh
```

This script will:
1. Download 4 LIBERO subsets from HuggingFace (`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`)
2. Download the VLM co-training data ([LLaVA-OneVision-COCO](https://huggingface.co/datasets/StarVLA/LLaVA-OneVision-COCO))
3. Create symlinks under `playground/Datasets/`
4. Copy `modality.json` into each dataset's `meta/` folder

<details>
<summary><b>Manual download (alternative)</b></summary>

```bash
# Download each dataset individually
huggingface-cli download IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot --repo-type dataset --local-dir playground/Datasets/LEROBOT_LIBERO_DATA/libero_spatial_no_noops_1.0.0_lerobot
huggingface-cli download IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot  --repo-type dataset --local-dir playground/Datasets/LEROBOT_LIBERO_DATA/libero_object_no_noops_1.0.0_lerobot
huggingface-cli download IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot    --repo-type dataset --local-dir playground/Datasets/LEROBOT_LIBERO_DATA/libero_goal_no_noops_1.0.0_lerobot
huggingface-cli download IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot      --repo-type dataset --local-dir playground/Datasets/LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot

# Copy modality.json to each subset
for d in playground/Datasets/LEROBOT_LIBERO_DATA/*/; do
  cp examples/LIBERO/train_files/modality.json "$d/meta/"
done
```
</details>

After this step, your `playground/Datasets/` should look like:

```
playground/Datasets/
├── LEROBOT_LIBERO_DATA/
│   ├── libero_spatial_no_noops_1.0.0_lerobot/
│   │   ├── meta/
│   │   │   ├── modality.json        ← required
│   │   │   └── ...
│   │   └── ...
│   ├── libero_object_no_noops_1.0.0_lerobot/
│   ├── libero_goal_no_noops_1.0.0_lerobot/
│   └── libero_10_no_noops_1.0.0_lerobot/
└── LLaVA-OneVision-COCO/             ← VLM co-training data
```

### Verify Your Dataloader

To make sure the data can be loaded correctly:

```bash
python starVLA/dataloader/lerobot_datasets.py \
  --config_yaml examples/LIBERO/train_files/starvla_cotrain_libero.yaml
```

---

## 3. Prepare Pretrained Models

Download the base VLM to `playground/Pretrained_models/`:

```bash
# Qwen3-VL (recommended)
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct --local-dir playground/Pretrained_models/Qwen3-VL-4B-Instruct

# For FAST framework, use the action-extended version instead:
# huggingface-cli download StarVLA/Qwen3-VL-4B-Instruct-Action --local-dir playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action
```

See [Model Zoo](model_zoo.md) for all available base models and finetuned checkpoints.

---

## 4. Understanding the Training Config

The full config is at [`examples/LIBERO/train_files/starvla_cotrain_libero.yaml`](../examples/LIBERO/train_files/starvla_cotrain_libero.yaml). Each parameter is explained below, matching the actual YAML structure.

---

### Top-level parameters

```yaml
run_id: starvla                   # Unique run identifier, used to name the output directory
run_root_dir: playground/Checkpoints  # Root output directory for all runs
seed: 42                          # Random seed for reproducibility
wandb_entity: your_wandb_entity   # W&B entity name (team/username)
wandb_project: llavavla           # W&B project name
is_debug: false                   # Debug mode: when true, loads minimal data for fast validation
version_id: "0.21"                # Config schema version, used by apply_config_compat for legacy format compat
```

---

### Framework

```yaml
framework:
  name: QwenGR00T                # which VLA architecture to use (see table below)
  qwenvl:
    base_vlm: ./playground/Pretrained_models/Qwen3-VL-4B-Instruct  # pretrained VLM path
    attn_implementation: flash_attention_2  # attention impl: flash_attention_2 (recommended) | sdpa | eager
    vl_hidden_dim: 2048          # VLM hidden dim, passed to action head as cross-attention dim
  action_model:
    action_dim: 7                # action vector dim. LIBERO = 7 (x, y, z, roll, pitch, yaw, gripper)
    state_dim: 7                 # proprioceptive state dim, typically equals action_dim
    action_horizon: 8            # total action chunk length (current step + future steps)
```

> **`action_horizon` vs `future_action_window_size`**: VLA action chunking predicts a sequence of `action_horizon` actions. Step 1 is the current action (executed immediately); steps 2–N are future actions (for temporal smoothness). Thus `future_action_window_size = action_horizon - 1 = 7`.
>
> The YAML only specifies `action_horizon: 8`. `future_action_window_size` is auto-derived by `apply_config_compat()` (`share_tools.py:465-477`) — no need to write it manually. If both are present and inconsistent, `action_horizon` takes precedence.

StarVLA supports four framework variants — just change `framework.name`:

| Framework | Name | Description |
|-----------|------|-------------|
| StarVLA-OFT | `QwenOFT` | MLP action head, simple & fast |
| StarVLA-FAST | `QwenFAST` | Discrete action tokens, autoregressive |
| StarVLA-π | `QwenPI` | Flow-matching diffusion action head |
| StarVLA-GR00T | `QwenGR00T` | Dual-system: VLM (System 2) + Flow-matching (System 1) |

---

### Datasets — VLM co-training data

```yaml
datasets:
  vlm_data:
    dataset_py: vlm_datasets     # dataloader module (starVLA/dataloader/vlm_datasets.py)
    dataformat: llava_json       # data format: llava_json = LLaVA-format JSON files
    dataset_use: sharegpt4v_coco # VLM dataset name (registered in dataloader)
    eval_dataset: sharegpt4v_coco # eval dataset name
    data_flatten: false          # whether to flatten image tokens into the text sequence
    base_interval: 2             # VLM data sampling interval: 1 VLM batch every 2 steps (interleaved with VLA)
    max_pixels: 307200           # max image pixels (images larger than this are downscaled)
    min_pixels: 784              # min image pixels (images smaller than this are upscaled), i.e. 28×28
    model_max_length: 2048       # tokenizer max sequence length
    model_type: qwen2.5vl        # VLM type, controls processor initialization
    per_device_batch_size: 4     # per-GPU batch size for VLM data
```

### Datasets — VLA robot action data

```yaml
  vla_data:
    dataset_py: lerobot_datasets # LeRobot-format dataloader
    data_root_dir: playground/Datasets/LEROBOT_LIBERO_DATA  # dataset root directory
    data_mix: libero_all         # dataset mixture: libero_all = all 4 suites | libero_goal = single suite
    action_type: delta_qpos      # action type: delta_qpos = joint deltas | absolute_qpos = absolute positions
    sequential_step_sampling: False  # whether to sample frames sequentially (False = random sampling)
    CoT_prompt: "Your task is {instruction}. To identify the key objects for your task. Locate their bounding boxes in [x1,y1,x2,y2] format."
                                 # Chain-of-Thought prompt template, {instruction} replaced with task description
    per_device_batch_size: 16    # per-GPU batch size for VLA data
    load_all_data_for_training: true  # preload all data into memory at training start (faster training)
    video_backend: torchvision_av # video decode backend: torchvision_av | decord (use former for av1 codec)
```

The `data_mix` field selects which datasets to combine. These mixtures are defined in [`examples/LIBERO/train_files/data_registry/data_config.py`](../examples/LIBERO/train_files/data_registry/data_config.py):

```python
DATASET_NAMED_MIXTURES = {
    "libero_all": [                              # all 4 suites
        ("libero_object_no_noops_1.0.0_lerobot",  1.0, "libero_franka"),
        ("libero_goal_no_noops_1.0.0_lerobot",    1.0, "libero_franka"),
        ("libero_spatial_no_noops_1.0.0_lerobot",  1.0, "libero_franka"),
        ("libero_10_no_noops_1.0.0_lerobot",      1.0, "libero_franka"),
    ],
    "libero_goal": [                             # single suite
        ("libero_goal_no_noops_1.0.0_lerobot",    1.0, "libero_franka"),
    ],
}
```

Each tuple is `(dataset_dir_name, sampling_weight, robot_type)`.

---

### Trainer

```yaml
trainer:
  max_train_steps: 100000         # maximum training steps
  num_warmup_steps: 5000          # LR warmup steps
  save_interval: 5000             # save checkpoint every N steps
  eval_interval: 100              # log eval metrics to W&B every N steps
  learning_rate:
    base: 2.5e-05                 # default LR (used for modules not listed below)
    qwen_vl_interface: 1.0e-05    # VLM module LR (lower to protect pretrained knowledge)
    action_model: 1.0e-04         # action head LR (higher for faster convergence on new module)
  lr_scheduler_type: cosine_with_min_lr  # LR scheduler type (transformers get_scheduler)
  scheduler_specific_kwargs:
    min_lr: 1.0e-06               # minimum LR for cosine annealing
  freeze_modules: 'qwen_vl_interface'  # comma-separated module names to freeze (VLM backbone frozen here)
  loss_scale:
    vla: 1.0                      # action loss weight
    vlm: 0.1                      # VLM co-training loss weight (auxiliary task, lower weight)
  max_grad_norm: 1.0              # max norm for gradient clipping
  weight_decay: 0.0               # weight decay (L2 regularization), disabled here
  logging_frequency: 10           # print training logs to console every N steps
  gradient_clipping: 1.0          # gradient clipping threshold (equivalent to max_grad_norm)
  gradient_accumulation_steps: 4  # gradient accumulation steps (effective batch = per_device_batch × GPUs × this)
  gradient_checkpointing: true    # enable gradient checkpointing (saves VRAM, slightly slower)

  optimizer:
    name: AdamW                   # optimizer name
    betas: [0.9, 0.95]            # AdamW β parameters
    eps: 1.0e-08                  # AdamW ε parameter (numerical stability)
    weight_decay: 1.0e-08         # optimizer-level weight_decay (independent of trainer-level weight_decay)
```

---

## 5. Understanding the Training Script

The training script [`examples/LIBERO/train_files/run_libero_train.sh`](../examples/LIBERO/train_files/run_libero_train.sh) wraps around `accelerate launch`. Key variables to customize:

```bash
###########################################################################################
# === Modify these for your environment ===
Framework_name=QwenOFT              # QwenOFT | QwenFAST | QwenPI | QwenGR00T
freeze_module_list=''               # e.g. 'qwen_vl' to freeze VLM backbone
base_vlm=playground/Pretrained_models/Qwen3-VL-4B-Instruct
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all                 # or libero_goal for single suite
run_root_dir=./results/Checkpoints
run_id=my_first_libero_run          # unique experiment name
###########################################################################################
```

The script launches distributed training with DeepSpeed ZeRO-2:

```bash
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \                    # number of GPUs
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  ...
```

> **Note:** Command-line arguments override YAML config values. This lets you keep one base config and vary parameters per experiment.

---

## 6. Launch Training

```bash
# From the repository root
bash examples/LIBERO/train_files/run_libero_train.sh
```

### What to expect

- Checkpoints are saved to `results/Checkpoints/{run_id}/checkpoints/`
- Training logs go to W&B (set `WANDB_MODE=disabled` to skip)
- The script copies itself to the output directory for reproducibility
- With 8× A100/H800 GPUs, training on `libero_all` takes roughly 30K steps (~10 epochs)

### Adjusting GPU count

Edit `--num_processes` in the script and `num_processes` in `starVLA/config/deepseeds/deepspeed_zero2.yaml` to match your available GPUs.

---

## 7. Evaluate a Pretrained Checkpoint

Evaluation uses a **client-server architecture**: a policy server (in the `starVLA` env) serves model inference, while the simulation client (in a separate `LIBERO` env) sends observations and receives actions.

### Step 0: Set up the LIBERO environment

Create a separate conda environment for the LIBERO simulator:

```bash
conda create -n libero python=3.10 -y
conda activate libero
pip install mujoco==3.2.3

# Clone and install LIBERO
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO && pip install -e . && cd ..

# Install eval dependencies
pip install tyro matplotlib mediapy websockets msgpack numpy==1.24.4
```

Or use the provided script:

```bash
bash examples/LIBERO/eval_files/install_libero.sh
```

### Step 1: Download a checkpoint

Download a pretrained checkpoint from [🤗 StarVLA/bench-libero](https://huggingface.co/collections/StarVLA/bench-libero):

```bash
huggingface-cli download StarVLA/Qwen3-VL-OFT-LIBERO-4in1 \
  --local-dir playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-LIBERO-4in1
```

### Step 2: Start the policy server (Terminal 1 — starVLA env)

Edit the checkpoint path in [`examples/LIBERO/eval_files/run_policy_server.sh`](../examples/LIBERO/eval_files/run_policy_server.sh):

```bash
CKPT=playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt
```

Then launch:

```bash
conda activate starVLA
bash examples/LIBERO/eval_files/run_policy_server.sh
```

Wait until you see `server listening on 0.0.0.0:6694`.

### Step 3: Run evaluation (Terminal 2 — LIBERO env)

Edit the paths in [`examples/LIBERO/eval_files/eval_libero.sh`](../examples/LIBERO/eval_files/eval_libero.sh), then:

```bash
conda activate libero
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
bash examples/LIBERO/eval_files/eval_libero.sh
```

### What to expect

- Each task runs 50 episodes by default
- Videos are saved under `results/{task_suite}/{checkpoint_name}/`
- Success rates are printed at the end

> **Tip:** To evaluate your own trained checkpoint, just point `CKPT` to your checkpoint under `results/Checkpoints/{run_id}/checkpoints/steps_XXXXX_pytorch_model.pt`.

---

## Next Steps

- **Try a different framework:** Change `Framework_name` in the training script to `QwenGR00T`, `QwenPI`, or `QwenFAST`
- **Explore other benchmarks:** Check out [SimplerEnv](../examples/SimplerEnv/), [RoboTwin](../examples/Robotwin/), [Calvin](../examples/calvin/), or [Behavior-1K](../examples/Behavior/)
- **World-Model-for-Action:** Use video-generation models (Cosmos, Wan) as action backbones — see [WM4A](WM4A.md)
- **Co-train with VLM data:** Learn how multi-objective training works in [CoTrainVLM](../examples/CoTrainVLM/)
- **Deploy on real robots:** See the [Franka example](../examples/Franka/) for real-world deployment
- **RL post-training:** Check [StarVLA × RLinf](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/starvla.html)
