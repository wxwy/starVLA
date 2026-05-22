

# Triton 需要 libcuda.so.1，部分容器/云环境需手动指定路径
# 若你的环境 libcuda.so 已在标准路径则可移除此行
export LD_LIBRARY_PATH=/opt/orion/orion_runtime/gpu/cuda:$LD_LIBRARY_PATH

# Triton 编译 CUDA utils (.c) 时需要 Python.h；conda venv 的头文件路径可能不在 sysconfig 返回的位置
# gcc 用 C_INCLUDE_PATH（不是 CPLUS_INCLUDE_PATH，后者仅 g++ 使用）
export C_INCLUDE_PATH=/root/miniconda3/envs/py310/include/python3.10:$C_INCLUDE_PATH

# 自动检测 NCCL 网络接口：bond0/ib0 优先（多节点 HPC），否则选第一个非 loopback 接口，兜底用 lo
if ip link show bond0 &>/dev/null; then
  export NCCL_SOCKET_IFNAME=bond0
  export NCCL_IB_HCA=mlx5_2,mlx5_3
elif ip link show eth0 &>/dev/null; then
  export NCCL_SOCKET_IFNAME=eth0
else
  export NCCL_SOCKET_IFNAME=lo
fi

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000  # timeout set to 1 hour (unit: seconds)
export NCCL_SOCKET_TIMEOUT_MS=360000
###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenGR00T
freeze_module_list='qwen_vl_interface'
base_vlm=playground/Pretrained_models/Qwen3-VL-4B-Instruct
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all
run_root_dir=./playground/Checkpoints
run_id=1229_libero4in1_qwen3oft
enable_local_checkpoint_staging=True
local_checkpoint_root=/tmp/nvme/starvla_ckpt
local_checkpoint_keep_count=2
gradient_accumulation_steps=2
num_processes=2
save_checkpoint_as_directory=True
save_with_training_state=False
checkpoint_max_shard_size=4GB
save_format=safetensors
num_workers=4
prefetch_factor=2
# === End of environment variable configuration ===
###########################################################################################


# export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
if [ "${enable_local_checkpoint_staging}" = "True" ] || [ "${enable_local_checkpoint_staging}" = "true" ]; then
  mkdir -p ${local_checkpoint_root}
fi
# mv this script to the output dir
cp $0 ${output_dir}/


if [ -n "${NUM_PROCESSES}" ]; then
  num_processes=${NUM_PROCESSES}
fi

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${num_processes} \
  --gradient_accumulation_steps ${gradient_accumulation_steps} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${libero_data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 8 \
  --datasets.vla_data.num_workers ${num_workers} \
  --datasets.vla_data.prefetch_factor ${prefetch_factor} \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 80000 \
  --trainer.save_interval 500 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --trainer.enable_local_checkpoint_staging ${enable_local_checkpoint_staging} \
  --trainer.local_checkpoint_root ${local_checkpoint_root} \
  --trainer.local_checkpoint_keep_count ${local_checkpoint_keep_count} \
  --trainer.save_checkpoint_as_directory ${save_checkpoint_as_directory} \
  --trainer.save_with_training_state ${save_with_training_state} \
  --trainer.checkpoint_max_shard_size ${checkpoint_max_shard_size} \
  --trainer.save_format ${save_format} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Libero \
  --wandb_entity silencewx-harbin-institute-of-technology \
  --trainer.is_resume True \                # 取消注释以从最新 checkpoint 恢复训练
  # --is_debug True



##### Multi-Server Multi-GPU training script #####
  # accelerate launch \
  #   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  #   --main_process_ip $MASTER_ADDR \
  #   --main_process_port $MASTER_PORT \
  #   --machine_rank $SLURM_PROCID \
  #   --num_machines $SLURM_NNODES \
  #   --num_processes=${TOTAL_GPUS} \
  #   starVLA/training/train_starvla.py \
  #   --config_yaml ${config_yaml} \
  #   --framework.name ${Framework_name} \
  #   --framework.qwenvl.base_vlm ${base_vlm} \
  #   --run_root_dir ${run_root_dir} \
  #   --run_id ${run_id} \
  #   --wandb_project your_project \
  #   --wandb_entity your_name
##### Multi-Server Multi-GPU training script #####
