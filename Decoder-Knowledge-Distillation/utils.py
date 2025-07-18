import torch
import torch.nn.functional as F
import math

# 计算前向kl散度
def compute_fkl(
        logits, 
        teacher_logits, 
        target, 
        padding_id,
        reduction="mean",
        temp = 2.0
    ):
        logits = logits / temp
        teacher_logits = teacher_logits / temp

        log_probs = torch.log_softmax(logits, -1, dtype=torch.float32)
        teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
        teacher_log_probs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
        kl = (teacher_probs * (teacher_log_probs - log_probs)) 
        kl = kl.sum(-1)
        if reduction == "mean":
            pad_mask = target.eq(padding_id)
            kl = kl.masked_fill_(pad_mask, 0.0)
            num_valid = (~pad_mask).sum()
            kl = kl.sum() / num_valid.clamp(min=1)

        return kl
# 计算反向kl散度
def compute_rkl(
        logits, 
        teacher_logits, 
        target, 
        padding_id,
        reduction="mean", 
        temp = 2.0
    ):
        logits = logits / temp
        teacher_logits = teacher_logits / temp

        probs = torch.softmax(logits, -1, dtype=torch.float32)
        log_probs = torch.log_softmax(logits, -1, dtype=torch.float32)
        teacher_log_probs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
        kl = (probs * (log_probs - teacher_log_probs))
        kl = kl.sum(-1)
        if reduction == "mean":
            pad_mask = target.eq(padding_id)
            kl = kl.masked_fill_(pad_mask, 0.0)
            num_valid = (~pad_mask).sum()
            kl = kl.sum() / num_valid.clamp(min=1)
        return kl

# 计算偏向前kl散度
def compute_skewed_fkl(
        logits, 
        teacher_logits, 
        target, 
        padding_id, 
        reduction="mean", 
        temp = 2.0,
        skew_lambda = 0.1
    ):
        logits = logits / temp
        teacher_logits = teacher_logits / temp

        probs = torch.softmax(logits, -1, dtype=torch.float32)
        teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
        mixed_probs = skew_lambda * teacher_probs + (1 - skew_lambda) * probs
        mixed_log_probs = torch.log(mixed_probs.clamp(min=1e-9))
        teacher_log_probs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
        kl = (teacher_probs * (teacher_log_probs - mixed_log_probs))
        kl = kl.sum(-1)
        if reduction == "mean":
            pad_mask = target.eq(padding_id)
            kl = kl.masked_fill_(pad_mask, 0.0)
            num_valid = (~pad_mask).sum()
            kl = kl.sum() / num_valid.clamp(min=1)
            
        return kl
# 计算偏向反kl散度    
def compute_skewed_rkl(
    logits, 
    teacher_logits, 
    target,
    padding_id,
    reduction="mean", 
    temp = 2.0,
    skew_lambda = 0.1
):
    logits = logits / temp
    teacher_logits = teacher_logits / temp
    
    probs = torch.softmax(logits, -1, dtype=torch.float32)
    teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
    mixed_probs = (1 - skew_lambda) * teacher_probs + skew_lambda * probs
    mixed_log_probs = torch.log(mixed_probs)
    log_probs = torch.log_softmax(logits, -1, dtype=torch.float32)
    kl = (probs * (log_probs - mixed_log_probs))
    kl = kl.sum(-1)
    
    if reduction == "mean":
        pad_mask = target.eq(padding_id)
        kl = kl.masked_fill_(pad_mask, 0.0)
        num_valid = (~pad_mask).sum()
        kl = kl.sum() / num_valid.clamp(min=1)

    return kl

def compute_jskl(
    student_logits, 
    teacher_logits, 
    labels=None, 
    beta=0.5, 
    temperature=2.0
):
    # Temperature scaling
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature

    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    # Log of mixture distribution
    log_beta = math.log(beta)
    log_1_minus_beta = math.log(1 - beta)
    mixture_log_probs = torch.logsumexp(
        torch.stack([student_log_probs + log_beta, teacher_log_probs + log_1_minus_beta]),
        dim=0,
    )

    # KL divergences
    kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
    kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)

    # Combine
    jsd = beta * kl_teacher + (1 - beta) * kl_student  # (B, L, V)
    jsd = jsd.sum(-1)  # (B, L)

    if labels is not None:
        mask = labels.eq(-100)
        jsd = jsd.masked_fill(mask, 0.0)
        num_valid = (~mask).sum()
        jsd = jsd.sum() / num_valid.clamp(min=1)
    else:
        jsd = jsd.mean()

    return jsd