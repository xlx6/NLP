import json
from tqdm import tqdm 
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch.utils.data import DataLoader
import torch

data = load_dataset('glue', 'sst2')
test_data = data['validation']

prompt_template = """
Below is an instruction that describes a task, paired with 
an input that provides further context. Write a response that appropriately 
completes the request. 
Before answering, think carefully about the question and create a step-by
step chain of thoughts to ensure a logical and accurate response. 
### Instruction: 
Analyze the given text from an online review and determine the sentiment 
polarity. Return a single number of either 0 and 1, with 0 being negative 
and 1 being the positive sentiment.  
### Input: 
{sentence} 
## Response:
<think/>
"""

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="fp4",
    bnb_4bit_use_double_quant=False
)


def evaluate(model, tokenizer, batch_size=8):
    results = []

    def collate_fn(batch):
        texts = []
        original_sentences = []
        labels = []
        for item in batch:
            prompt = prompt_template.format(sentence=item['sentence'])
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )
            texts.append(text)
            original_sentences.append(item['sentence'])
            labels.append(item['label'])
        return texts, original_sentences, labels

    dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    for texts, original_sentences, labels in tqdm(dataloader, desc="Evaluating"):
        model_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
                do_sample=False
            )

        for i in range(len(texts)):
            output_ids = generated_ids[i][len(model_inputs.input_ids[i]):].tolist()
            try:
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0

            thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

            results.append({
                "sentence": original_sentences[i],
                "label": labels[i],
                "predict": content,
                "think": thinking_content
            })
            print(results[-1])

    with open("Qwen3-14B-Instruct.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

model_name = "/root/autodl-fs/models/Qwen/Qwen3-14B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype= torch.bfloat16
)
evaluate(model, tokenizer)