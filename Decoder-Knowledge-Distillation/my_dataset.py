from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from datasets import load_dataset

class AlpacaPromptDataset(Dataset):
    def __init__(self, dataset, tokenizer: PreTrainedTokenizer, max_length=512, split='train'):
        # if dataset_name_or_path.endswith(".json"):
        #     self.dataset = load_dataset("json", data_files=dataset_name_or_path, split=split)
        # else:
        self.dataset = dataset

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 
                        ### Instruction: 
                        {} 
                        ### Input: 
                        {} 
                        ### Response: 
                        {}"""

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        instruction = data.get("instruction", "")
        input_text = data.get("input", "")
        output = data.get("output", "")

        full_text = self.alpaca_prompt.format(instruction, input_text, output)
        prompt_text = self.alpaca_prompt.format(instruction, input_text, "")  

        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        # 只计算 Response 部分
        prompt_encoding = self.tokenizer(
            prompt_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        prompt_len = (prompt_encoding["input_ids"] != self.tokenizer.pad_token_id).sum().item()

        labels = encoding["input_ids"].clone()
        labels[0, :prompt_len] = -100

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": labels,
        }
