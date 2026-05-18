# 模型库

我们发布了一系列修改后的模型和微调检查点，以便复现和下游使用。

## 可用修改模型

| 模型 | 描述 | 链接 |
| --- | --- | --- |
| **Qwen2.5-VL-3B-Action** | 扩展 Qwen2.5-VL 词表，加入 Fast Tokens | [🤗 Hugging Face](https://huggingface.co/StarVLA/Qwen2.5-VL-3B-Instruct-Action) |
| **Qwen3-VL-4B-Action** | 扩展 Qwen3-VL 词表，加入 Fast Tokens | [🤗 Hugging Face](https://huggingface.co/StarVLA/Qwen3-VL-4B-Instruct-Action) |

## 可用微调检查点

| 模型 | 描述 | WidowX | 链接 |
| --- | --- | --- | --- |
| **QWen2.5-FAST-Bridge-RT-1** | 在 [Bridge](https://huggingface.co/datasets/IPEC-COMMUNITY/bridge_orig_lerobot) 和 [Fractal](https://huggingface.co/datasets/IPEC-COMMUNITY/fractal20220817_data_lerobot) 上训练 | 58.6 | [🤗 Hugging Face](https://huggingface.co/StarVLA/Qwen-FAST-Bridge-RT-1) |
| **QWen2.5-OFT-Bridge-RT-1** | 在 [Bridge](https://huggingface.co/datasets/IPEC-COMMUNITY/bridge_orig_lerobot) 和 [Fractal](https://huggingface.co/datasets/IPEC-COMMUNITY/fractal20220817_data_lerobot) 上训练 | 41.8 | [🤗 Hugging Face](https://huggingface.co/StarVLA/Qwen-OFT-Bridge-RT-1) |
| **QWen2.5-PI-Bridge-RT-1** | 在 [Bridge](https://huggingface.co/datasets/IPEC-COMMUNITY/bridge_orig_lerobot) 和 [Fractal](https://huggingface.co/datasets/IPEC-COMMUNITY/fractal20220817_data_lerobot) 上训练 | 62.5 | [🤗 Hugging Face](https://huggingface.co/StarVLA/Qwen-FM-Bridge-RT-1) |
| **QWen2.5-GR00T-Bridge-RT-1** | 在 [Bridge](https://huggingface.co/datasets/IPEC-COMMUNITY/bridge_orig_lerobot) 和 [Fractal](https://huggingface.co/datasets/IPEC-COMMUNITY/fractal20220817_data_lerobot) 上训练 | 63.6 | [🤗 Hugging Face](https://huggingface.co/StarVLA/Qwen-PI-Bridge-RT-1) |
| **QWen-GR00T-Bridge** | 仅在 [Bridge](https://huggingface.co/datasets/IPEC-COMMUNITY/bridge_orig_lerobot) 上训练 | 71.4 | [🤗 Hugging Face](https://huggingface.co/StarVLA/Qwen-GR00T-Bridge) |
| **QWen3VL-OFT-Bridge-RT-1** | 在 [Bridge](https://huggingface.co/datasets/IPEC-COMMUNITY/bridge_orig_lerobot) 和 [Fractal](https://huggingface.co/datasets/IPEC-COMMUNITY/fractal20220817_data_lerobot) 上训练 | 42.7 | [🤗 Hugging Face](https://huggingface.co/StarVLA/Qwen3VL-OFT-Bridge-RT-1) |
| **QWen3VL-GR00T-Bridge-RT-1** | 在 [Bridge](https://huggingface.co/datasets/IPEC-COMMUNITY/bridge_orig_lerobot) 和 [Fractal](https://huggingface.co/datasets/IPEC-COMMUNITY/fractal20220817_data_lerobot) 上训练 | 65.3 | [🤗 Hugging Face](https://huggingface.co/StarVLA/Qwen3VL-GR00T-Bridge-RT-1) |
| **QWen3VL-PI_v3-Bridge-RT-1** | 在 [Bridge](https://huggingface.co/datasets/IPEC-COMMUNITY/bridge_orig_lerobot) 和 [Fractal](https://huggingface.co/datasets/IPEC-COMMUNITY/fractal20220817_data_lerobot) 上训练 | 69.8 | [🤗 Hugging Face](https://huggingface.co/StarVLA/Qwen3VL-PI_v3-Bridge-RT_1) |

| 模型 | 描述 | 平均长度 | 链接 |
| --- | --- | --- | --- |
| **QWen2.5VL-GR00T-Calvin_D_D** | 在 [Calvin_D_D](https://github.com/EmbodiedAI-RoboTron/RoboTron-Mani/tree/lerobot/examples/calvin) 上训练 | 3.786 | [🤗 Hugging Face](https://huggingface.co/Simplicissimus-S/StarVLA-QwenGR00T_Qwen2.5-VL-3B-Instruct-Action_calvin_D_D) |
