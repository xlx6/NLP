import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput

from transformers import TrainerCallback
import math
import swanlab

class AlphaScheduler(TrainerCallback):
    def __init__(self, total_steps, warmup_ratio=0.1):
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.non_warmup_steps = max(1, total_steps - self.warmup_steps)
        self.v_midpoint = self.warmup_steps + self.non_warmup_steps // 2
        self.v_descent_steps = max(1, self.v_midpoint - self.warmup_steps)
        self.v_ascent_steps = max(1, self.total_steps - self.v_midpoint)

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step

        if step > self.total_steps:
            return

        scheduler_type = args.hp_method.lower()
        assert scheduler_type in ['cosine_growth', 'cosine_vshape', 'cosine_descent', 'cosine_^shape']

        if scheduler_type == 'cosine_growth':
            new_alpha = self._cosine_growth_schedule(step)
        elif scheduler_type == 'cosine_vshape':
            new_alpha = self._vshape_schedule(step)
        elif scheduler_type == 'cosine_descent':
            new_alpha = self._cosine_descent_schedule(step)
        else:
            new_alpha = self._cosine_growth_descent_schedule(step)

        args.alpha = new_alpha
        if step % 10 == 0:
            swanlab.log({"alpha": new_alpha})

    def _cosine_growth_schedule(self, step):
        if step <= self.warmup_steps:
            return 0.0
        else:
            progress = (step - self.warmup_steps) / self.non_warmup_steps
            return 1 - math.cos(progress * (math.pi / 2))

    def _vshape_schedule(self, step):
        if step <= self.warmup_steps:
            return 1.0
        elif step <= self.v_midpoint:
            progress = (step - self.warmup_steps) / self.v_descent_steps
            return math.cos(progress * (math.pi / 2))
        else:
            progress = (step - self.v_midpoint) / self.v_ascent_steps
            return math.sin(progress * (math.pi / 2))

    def _cosine_descent_schedule(self, step):
        if step <= self.warmup_steps:
            return 1.0
        else:
            progress = (step - self.warmup_steps) / self.non_warmup_steps
            return math.cos(progress * (math.pi / 2))

    def _cosine_growth_descent_schedule(self, step):
        """alpha: 0->1->0"""
        if step <= self.warmup_steps:
            return 0.0
        elif step <= self.v_midpoint:
            progress = (step - self.warmup_steps) / self.v_descent_steps
            return 1 - math.cos(progress * (math.pi / 2))
        else:
            progress = (step - self.v_midpoint) / self.v_ascent_steps
            return 1 - math.sin(progress * (math.pi / 2))


class DistillationTrainingArguments(TrainingArguments):
    """
    parameters:
        alpha(float): CE loss wight
        beta(float): hidden distill weight
        gamma(float): logits distill weight
        distill_layers(List[int]): index of selected distill layers
    """
    def __init__(self, *args,
                 alpha=0.5,       
                 beta=0.3,          
                 gamma=0.2,           
                 temperature=2.0,    
                 distill_layers=None,  
                 hp_method=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature
        self.distill_layers = distill_layers
        self.hp_method = hp_method

            
class DistillationTrainer(Trainer): 
        def __init__(self, *args, teacher_model=None, **kwargs): 
            super().__init__(*args, **kwargs) 
            self.teacher = teacher_model 
            self.teacher.to(self.model.device)
            self.teacher.eval() 
            for param in self.teacher.parameters():
                param.requires_grad = False
            
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None): 
            labels = inputs.get("labels", None)
            # student outputs
            outputs_student = model(**inputs, output_hidden_states=True) 
            student_logits = outputs_student.logits
            student_hidden_states = outputs_student.hidden_states
            # ([bsz, seq_len, d_model]......)

            
            # print(len(student_hidden_state))
            # print(student_hidden_state[0].shape)
            
            student_ce_loss = outputs_student.loss 

            
            with torch.no_grad(): 
                outputs_teacher = self.teacher(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                    output_hidden_states=True
                )
            teacher_logits = outputs_teacher.logits
            teacher_hidden_states = outputs_teacher.hidden_states
            # print(len(outputs_teacher.hidden_states))
            # print(outputs_teacher.hidden_states[0].shape)
            # return None
             
            temperature = self.args.temperature
            loss_function = nn.KLDivLoss(reduction="batchmean") 
            loss_logits = (loss_function( 
                F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1), 
                F.softmax(outputs_teacher.logits / self.args.temperature, dim=-1)
            ) * (temperature ** 2)) 
            loss_hidden = 0.0
            if self.args.distill_layers:
                mse_loss_fn = nn.MSELoss()
                for idx in self.args.distill_layers:
                    s_hidden = student_hidden_states[idx][:, 0, :]
                    t_hidden = teacher_hidden_states[2*idx][:, 0, :]
                    # [bsz, d_model]
    
                    s_hidden = F.normalize(s_hidden, p=2, dim=-1)
                    t_hidden = F.normalize(t_hidden, p=2, dim=-1)
                    loss_hidden += mse_loss_fn(s_hidden, t_hidden)
            # print("=====================================================")
            # print(f"labels: {labels.shape}")
            # print(f"logits: {student_logits.shape}")
            loss = self.args.alpha * student_ce_loss + (1. - self.args.alpha) * (loss_logits + loss_hidden)
            return (loss, outputs_student) if return_outputs else loss

    

            

