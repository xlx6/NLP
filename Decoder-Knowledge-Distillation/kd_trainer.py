from trl import SFTTrainer
import torch
import torch.nn.functional as F
from utils import (
    compute_fkl,
    compute_rkl,
    compute_skewed_fkl,
    compute_skewed_rkl,
    compute_jskl
)

class KDTrainer(SFTTrainer):
    def __init__(self, *args, teacher_model=None, kd_type="none", use_ce=None, temperature=2.0, alpha=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.kd_type = kd_type
        self.temperature = temperature
        self.alpha = alpha
        self.use_ce = use_ce

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        outputs = model(**inputs)
        student_logits = outputs.logits
        

        # student CE loss
        loss_ce = outputs.loss
        if loss_ce is None:
            raise ValueError("Student model outputs.loss is None. Please ensure labels are correctly passed in and model returns loss.")

        if self.kd_type == "none":
            return (loss_ce, outputs) if return_outputs else loss_ce

        # 蒸馏
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits
        # assert student_logits.shape[:-1] == teacher_logits.shape[:-1], \
        #     f"Shape mismatch: student {student_logits.shape}, teacher {teacher_logits.shape}"

        # 对齐长度（防止 student/teacher vocab 不一致）
        if student_logits.shape[-1] != teacher_logits.shape[-1]:
            #print(f"[Warning] Student vocab size {student_logits.shape[-1]} != Teacher vocab size {teacher_logits.shape[-1]}. Truncating teacher logits.")
            teacher_logits = teacher_logits[:, :, :student_logits.shape[-1]]

        # KL蒸馏
        if self.kd_type == "fkl":
            kl_loss = compute_fkl(student_logits, teacher_logits, labels, padding_id=-100, temp=self.temperature) 
        elif self.kd_type == "rkl":
            kl_loss = compute_rkl(student_logits, teacher_logits, labels, padding_id=-100, temp=self.temperature)
        elif self.kd_type == "seqkd":
            teacher_pred = torch.argmax(teacher_logits, dim=-1)
            kl_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                teacher_pred.view(-1),
                ignore_index=-100
            )
        elif self.kd_type == 'jskl':
            kl_loss = compute_jskl(student_logits, teacher_logits, labels, temperature = self.temperature)
        else:
            raise ValueError(f"Unsupported kd_type: {self.kd_type}")

        if self.use_ce:
            total_loss = self.alpha * kl_loss + (1 - self.alpha) * loss_ce
        else:
            total_loss = kl_loss
            
        return (total_loss, outputs) if return_outputs else total_loss
