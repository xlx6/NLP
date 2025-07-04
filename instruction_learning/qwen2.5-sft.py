from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset
import torch
import swanlab
import argparse
from utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1, 
                       help="Number of training samples to use")
    parser.add_argument("--n", type=int, default=None, 
                       help="Number of training samples to use")
    parser.add_argument("--bsz", type=int, default=4, 
                       help="Number of training samples to use")
    parser.add_argument("--gas", type=int, default=4, 
                       help="Number of training samples to use")
    return parser.parse_args()

def main(args):
    n = args.n
    swanlab.login(api_key="3ImGDbNAVVcnvp5J3sagY", save=True)
    swanlab.init(
        project="Instruction-Learning-Qwen2.5-0.5B",
        config={
            "num_samples": str(n) if n else 'all',
            "model": "Qwen2.5-1.5B",
        }
    )

    train_df, test_df = load_data(n)
    dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(test_df)
    
    model = AutoModelForCausalLM.from_pretrained("/root/autodl-fs/models/Qwen/Qwen2___5-0___5B-Instruct")
    
    # print(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM"
    )
    tokenizer = AutoTokenizer.from_pretrained("/root/autodl-fs/models/Qwen/Qwen2___5-0___5B-Instruct")
    model = get_peft_model(model, lora_config)
    role = "you are a sementic analysis expert."
    Instruction = "Analyze the given text from an online review and determine the sentiment polarity. Return a single number: 0 for negative sentiment, 1 for positive sentiment. Only output 0 or 1. Do not explain or add any extra text."
    
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['sentence'])):
            text = f"### Instruction: {Instruction}\n ### Input: {example['sentence'][i]}\n ### Answer: {example['label'][i]}"
            output_texts.append(role + '\n' + text)
        return output_texts
    
    response_template = " ### Answer:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    train_args = SFTConfig(
        output_dir="./model_checkpoints",
        report_to="swanlab",
        fp16=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.eval_bsz,
        gradient_accumulation_steps=args.gas,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=50,
    )
    trainer = SFTTrainer(
        model,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        args=train_args,
        formatting_func=formatting_prompts_func,
        data_collator=collator
    )
    
    trainer.train()
    
    swanlab.finish()

if __name__=='__main__':
    args = parse_args()
    main(args)