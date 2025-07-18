import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import argparse
import math

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", type=str, required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--save_name", type=str, required=True, help="Output filename (without .json)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation")
    parser.add_argument("--max_samples", type=int, default=300, help="Number of samples to generate (optional)")
    parser.add_argument("--base_params", type=float, default=None, help="Params of Base model")
    return parser.parse_args()

args = parse_args()

# 1. 加载模型
base_model_path = f"/root/autodl-fs/models/Qwen/Qwen2.5-{(lambda x: int(x) if x == int(x) else x)(args.base_params)}B-Instruct"
lora_model_path = args.lora_path

device = "cuda" if torch.cuda.is_available() else "cpu"

print("🔧 Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16, trust_remote_code=True)
model = PeftModel.from_pretrained(model, lora_model_path)
model = model.to(device)
model.eval()

# 2. 加载测试数据
with open("./data/test.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

test_data = test_data[:args.max_samples]

# 3. 构建 prompt 模板
alpaca_prompt = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request. \n\n"
    "### Instruction: \n{} \n\n"
    "### Input: \n{} \n\n"
    "### Response: \n"
)

# 4. 批量生成
outputs = []
batch_size = args.batch_size
num_batches = math.ceil(len(test_data) / batch_size)

print(f"🚀 Generating with batch size = {batch_size} ...")

for i in tqdm(range(num_batches), desc="Generating responses"):
    batch = test_data[i * batch_size: (i + 1) * batch_size]

    batch_prompts = [
        alpaca_prompt.format(sample["instruction"], sample["input"])
        for sample in batch
    ]

    encoded = tokenizer(
        batch_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded_outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    for j, sample in enumerate(batch):
        prompt_len = len(tokenizer.decode(encoded["input_ids"][j], skip_special_tokens=True))
        full_output = decoded_outputs[j]

        # 截取新生成部分
        gen_text = full_output[prompt_len:].strip()

        outputs.append({
            "instruction": sample["instruction"],
            "input": sample["input"],
            "reference_output": sample.get("output", ""),
            "generated_output": gen_text
        })

# 5. 保存结果
save_path = f"./generate/{args.save_name}.json"
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(outputs, f, indent=2, ensure_ascii=False)

print(f"✅ All generations saved to {save_path}")