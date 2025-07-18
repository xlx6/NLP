# python train_student.py \
#   --num_params 0.5 \
#   --kd_type rkl \
#   --epochs 3 \
#   --output_dir ./outputs/student/0.5B_rkl_step300

# python train_student.py \
#   --num_params 0.5 \
#   --kd_type fkl \
#   --epochs 3 \
#   --output_dir ./outputs/student/0.5B_fkl_step200

# python train_student.py \
#   --num_params 0.5 \
#   --kd_type seqkd \
#   --epochs 3 \
#   --output_dir ./outputs/student/0.5B_seqkl_step200

python train_student.py \
  --num_params 0.5 \
  --kd_type jskl \
  --epochs 3 \
  --output_dir ./outputs/student/0.5B_jskl_step200