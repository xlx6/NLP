import os
from unsloth import FastLanguageModel 
from unsloth import is_bfloat16_supported 
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from dataset import SFTDataset
import swanlab
from transformers import BitsAndBytesConfig
import torch

from datasets import load_dataset 
from trl import SFTTrainer 
from transformers import TrainingArguments 

from kd_trainer import KDTrainer
import argparse

os.environ['UNSLOTH_RETURN_LOGITS'] = '1'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train")
    parser.add_argument("--kd_type", type=str, default="fkl", help="kl loss function")
    parser.add_argument("--num_params", type=float, default=0.5, help="student trainable params")
    parser.add_argument("--output_dir", type=str, default='./outputs', help="student save path")
    return parser.parse_args()
    

def main():
    args = parse_args()
    max_seq_length = 1024
    dtype = None 
    load_in_4bit = True 
    teacher_lora_path = './outputs/teacher/checkpoint-500'
    teacher_base_path = '/root/autodl-fs/models/Qwen/Qwen2.5-7B-Instruct'
    student_base_path = f'/root/autodl-fs/models/Qwen/Qwen2.5-{str(args.num_params)}B-Instruct'
    
    swanlab.login(api_key="3ImGDbNAVVcnvp5J3sagY", save=True)
    swanlab.init(
        project="Decoder-Knowledge-Distillation",
        experiment_name=f"Student-{str(args.num_params)} kd_type-{args.kd_type} Model training-epochs-{args.epochs}",
        config={
            "model": f"Qwen2.5-{str(args.num_params)}B",
        }
    )
    tokenizer = AutoTokenizer.from_pretrained(teacher_base_path)

    # load Teacher
    teacher_model, _ = FastLanguageModel.from_pretrained(
        model_name = teacher_base_path,
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = load_in_4bit,
    ) 
    
    teacher_model = PeftModel.from_pretrained(
        teacher_model,
        teacher_lora_path,
        is_trainable=False
    )

    # load Student
    student, _ = FastLanguageModel.from_pretrained( 
        model_name = student_base_path, 
        max_seq_length = max_seq_length, 
        dtype = dtype, 
    ) 
    
    model = FastLanguageModel.get_peft_model( 
        student, 
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
    
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. \n\n### Instruction: \n{} \n\n### Input: \n{} \n\n### Response: \n{}"""

    EOS_TOKEN = tokenizer.eos_token 
    def formatting_prompts_func(examples): 
        instructions = examples["instruction"] 
        inputs       = examples["input"] 
        outputs      = examples["output"] 
        texts = [] 
        for instruction, input, output in zip(instructions, inputs, outputs): 
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN 
            texts.append(text) 
        return { "text" : texts, }


     

    train_dataset = load_dataset("json", data_files = "./data/train.json", split="train") 
    # eval_dataset = load_dataset("json", data_files = "./data/test.json", split="test")
    train_dataset = train_dataset.map(formatting_prompts_func, batched = True,) 
    # eval_dataset = eval_dataset.map(formatting_prompts_func, batched = True,)

    training_args = TrainingArguments( 
        per_device_train_batch_size = 2, 
        gradient_accumulation_steps = 4, 
        num_train_epochs = args.epochs,
        warmup_steps = 5, 
        max_steps=200,
        fp16 = not is_bfloat16_supported(), 
        bf16 = is_bfloat16_supported(), 
        save_strategy="steps",
        save_steps=100,
        logging_strategy = "steps",
        logging_steps=10,
        optim = "adamw_8bit", 
        weight_decay = 0.01, 
        lr_scheduler_type = "linear", 
        seed = 42, 
        output_dir = args.output_dir, 
        report_to = "swanlab", 
    ) 
    trainer = KDTrainer( 
        model = model, 
        teacher_model = teacher_model,
        kd_type = args.kd_type,
        use_ce = True,
        processing_class = tokenizer, 
        train_dataset = train_dataset,
        # eval_dataset = eval_dataset,
        dataset_text_field = "text", 
        max_seq_length = max_seq_length, 
        dataset_num_proc = 2, 
        packing = True, 
        args = training_args, 
    ) 
    trainer.train()

    swanlab.finish()

if __name__ == '__main__':
    main()