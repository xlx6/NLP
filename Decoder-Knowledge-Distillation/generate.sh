python batch_generate.py \
    --lora_path ./outputs/teacher/checkpoint-500 \
    --save_name teacher \
    --batch_size 32 \
    --max_samples 2048 \
    --base_params 7
    
python batch_generate.py \
    --lora_path ./outputs/student/0.5B_none/checkpoint-200 \
    --save_name 0_5B_none \
    --batch_size 256 \
    --max_samples 2048 \
    --base_params 0.5
    
python batch_generate.py \
    --lora_path ./outputs/student/0.5B_fkl/checkpoint-200 \
    --save_name 0_5B_fkl \
    --batch_size 256 \
    --max_samples 2048 \
    --base_params 0.5
    
python batch_generate.py \
    --lora_path ./outputs/student/0.5B_rkl/checkpoint-200 \
    --save_name 0_5B_rkl \
    --batch_size 256 \
    --max_samples 2048 \
    --base_params 0.5
    
python batch_generate.py \
    --lora_path ./outputs/student/0.5B_jskl/checkpoint-200 \
    --save_name 0_5B_jskl \
    --batch_size 256 \
    --max_samples 2048 \
    --base_params 0.5
    
python batch_generate.py \
    --lora_path ./outputs/student/0.5B_seqkl/checkpoint-200 \
    --save_name 0_5B_seqkl \
    --batch_size 256 \
    --max_samples 2048 \
    --base_params 0.5
