import os
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from dataset import SFTDataset
import swanlab
from transformers import BitsAndBytesConfig
import torch
from unsloth import FastLanguageModel 
from datasets import load_dataset 
from trl import SFTTrainer 
from transformers import TrainingArguments 
from unsloth import is_bfloat16_supported 

max_seq_length = 2048 
dtype = None 
load_in_4bit = True 

swanlab.login(api_key="3ImGDbNAVVcnvp5J3sagY", save=True)
swanlab.init(
    project="Decoder-Knowledge-Distillation",
    experiment_name=f"Teacher Model training",
    config={
        "model": "Qwen2.5-7B",
    }
)
model, tokenizer = FastLanguageModel.from_pretrained( 
    model_name = "/root/autodl-fs/models/Qwen/Qwen2.5-7B-Instruct", 
    max_seq_length = max_seq_length, 
    dtype = dtype, 
    load_in_4bit = load_in_4bit, 
) 

alpaca_prompt = """Below is an instruction that describes a task, paired with 
an input that provides further context. Write a response that appropriately 
completes the request. 

### Instruction: 
{} 

### Input: 
{} 

### Response: 
{}"""

EOS_TOKEN = tokenizer.eos_token 
# Must add EOS_TOKEN
 
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    
    input_texts = []
    label_ids = []

    for instruction, input_text, output in zip(instructions, inputs, outputs):
        prompt = alpaca_prompt.format(instruction, input_text, "")
        full_text = prompt + output + EOS_TOKEN

        full_tokens = tokenizer(full_text, truncation=True, max_length=max_seq_length, padding=False)
        prompt_tokens = tokenizer(prompt, truncation=True, max_length=max_seq_length, padding=False)

        input_ids = full_tokens["input_ids"]
        labels = input_ids.copy()

        prompt_len = len(prompt_tokens["input_ids"])
        labels[:prompt_len] = [-100] * prompt_len

        input_texts.append(input_ids)
        label_ids.append(labels)

    return {
        "input_ids": input_texts,
        "labels": label_ids,
    }
     

dataset = load_dataset("json", data_files = "./data/train.json", split="train") 
dataset = dataset.map(formatting_prompts_func, batched = True,) 



model = FastLanguageModel.get_peft_model( 
    model, 
    r = 8, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",], 
    lora_alpha = 32, 
    lora_dropout = 0, 
    bias = "none",    
    use_gradient_checkpointing = "unsloth", 
    random_state = 42, 
    use_rslora = False,  
    loftq_config = None, 
)


training_args = TrainingArguments( 
    per_device_train_batch_size = 2, 
    gradient_accumulation_steps = 4, 
    warmup_steps = 5, 
    max_steps = 600, 
    learning_rate = 5e-5, 
    fp16 = not is_bfloat16_supported(), 
    bf16 = is_bfloat16_supported(), 
    save_steps=300,
    logging_steps = 10, 
    optim = "adamw_8bit", 
    weight_decay = 0.01, 
    lr_scheduler_type = "linear", 
    seed = 42, 
    output_dir = "./outputs/teacher-responseonly", 
    report_to = "swanlab", 
) 
trainer = SFTTrainer( 
    model = model, 
    processing_class = tokenizer, 
    train_dataset = dataset, 
    dataset_text_field = "text", 
    max_seq_length = max_seq_length, 
    dataset_num_proc = 2, 
    packing = False, 
    args = training_args, 
) 
trainer.train() 
model.save_pretrained("qwen_teacher_finetune")  
# Local saving
 
tokenizer.save_pretrained("qwen_teacher_finetune") 

swanlab.finish()


