import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm  # ✅ 加载进度条库
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", type=str, default=None, help="Number of epochs to train")
    parser.add_argument("--save_name", type=str, default=None, help="Number of epochs to train")

    return parser.parse_args()

args = parse_args()

# 1. 加载模型（base + LoRA 权重）
base_model_path = "/root/autodl-fs/models/Qwen/Qwen2.5-7B-Instruct"  # 修改为你的base模型路径
lora_model_path = args.lora_path  # 修改为你的LoRA权重路径

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16, trust_remote_code=True)
model = PeftModel.from_pretrained(model, lora_model_path)
model = model.to(device)
model.eval()

# 2. 读取 Alpaca 的 test.json（你的测试集）
with open("./data/test.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

# 3. 应用 Alpaca 模板并生成响应
alpaca_prompt = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request. \n\n"
    "### Instruction: \n{} \n\n"
    "### Input: \n{} \n\n"
    "### Response: \n"
)

outputs = []
for i, example in enumerate(tqdm(test_data[:300], desc="Generating responses")):  # ✅ 加 tqdm 进度条
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output_ref = example.get("output", "")

    prompt = alpaca_prompt.format(instruction, input_text, "")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    output_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    outputs.append({
        "instruction": instruction,
        "input": input_text,
        "reference_output": output_ref,
        "generated_output": output_text.strip()
    })

# 4. 保存响应等信息到 outputs.json
with open(f"./generate/{args.save_name}.json", "w", encoding="utf-8") as f:
    json.dump(outputs, f, indent=2, ensure_ascii=False)

print("✅ All generations saved to outputs.json")
