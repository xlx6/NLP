python train_teacher_lora.py \
  --teacher_model_path /root/autodl-fs/models/Qwen/Qwen2.5-0.5B-Instruct \
  --output_dir ./outputs/teacher_lora \
  --kd_type none \
  --dataset_name yahma/alpaca-cleaned \
  --num_train_epochs 10