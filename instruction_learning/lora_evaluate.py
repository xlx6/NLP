from datasets import load_dataset
import transformers
import torch
import json
import os
from tqdm import tqdm
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Load SST-2 dataset
data = load_dataset('glue', 'sst2')
dataset = data['validation']

# Define multiple models to evaluate
ttt = "/root/autodl-fs/models/Qwen/Qwen2___5-7B-Instruct"
# models = [
#     "/root/autodl-fs/models/Qwen/Qwen2___5-7B-Instruct",
#     "/root/autodl-fs/models/Qwen/Qwen2___5-0___5B-Instruct"
# ]
models = [
    './model_checkpoints256/256-samples-checkpoint-40',
    './model_checkpoints64/64-samples-checkpoint-24',
    './model_checkpoints16/16-samples-checkpoint-6'
]

# Create output directory if it doesn't exist
output_dir = "model_evaluation_results"
os.makedirs(output_dir, exist_ok=True)

# Prompt template
prompt_style = """
you are a sementic analysis expert.

### Instruction:
Analyze the given text from an online review and determine the sentiment polarity. Return a single number: 0 for negative sentiment, 1 for positive sentiment. Only output 0 or 1. Do not explain or add any extra text.

### Input:
{sentence}

### Response:
"""

def evaluate_model(model_id):
    print(f"Evaluating model: {model_id}")
    
    # 加载 LoRA 配置
    peft_config = PeftConfig.from_pretrained(model_id)
    base_model_path = peft_config.base_model_name_or_path

    # 加载 base model 和 tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # 加载 LoRA adapter
    model = PeftModel.from_pretrained(base_model, model_id)
    model.eval()

    # 创建文本生成 pipeline
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=10,
        return_full_text=False
    )

    batch_size = 64
    all_inputs = [prompt_style.format(sentence=item['sentence']) for item in dataset]
    results = []

    # 批量推理
    for i in tqdm(range(0, len(all_inputs), batch_size)):
        batch_inputs = all_inputs[i:i+batch_size]
        batch_items = dataset[i:i+batch_size]
        responses = pipeline(batch_inputs)
        for j, response in enumerate(responses):
            generated_text = response[0]["generated_text"]
            result = {
                "sentence": batch_items['sentence'][j],
                "label": int(batch_items['label'][j]),
                "response": generated_text,
                "model_id": model_id
            }
            results.append(result)
            print(results[-1])
    return results


# Evaluate each model and save results
for model_id in models:
    # Get model name for the filename
    model_name = model_id.split("/")[-1]
    
    # Evaluate the model
    results = evaluate_model(model_id)
    
    # Save results to JSON
    output_file = os.path.join(output_dir, f"{model_name}_noquantity.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results for {model_id} saved to {output_file}")

print("Evaluation complete for all models.")
