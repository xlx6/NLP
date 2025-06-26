import os
import sys
import logging
import datasets
import evaluate

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

from transformers import (
    BertTokenizerFast, 
    DataCollatorWithPadding,
    DebertaV2TokenizerFast,
    DebertaV2Tokenizer,
    Trainer, 
    TrainingArguments, 
    DebertaV2Model,
    DebertaV2PreTrainedModel
    )
from peft import get_peft_model, LoraConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from bs4 import BeautifulSoup
from tqdm import tqdm
import re
import wandb
import evaluate
import random

# os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_lora_config():
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32, 
        lora_dropout=0.1, 
        target_modules=["query_proj", "value_proj", "dense"],
        bias="none",  
    )
    return lora_config

def extract_and_save_hidden_states(model, dataloader, save_path='./train_hidden.csv', device='cuda'):
    model.eval()
    all_hidden = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting hidden states"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            cls_output = outputs.hidden_states[-1][:, 0]

            all_hidden.append(cls_output.detach().cpu())
            all_labels.append(labels.detach().cpu())

    all_hidden = torch.cat(all_hidden, dim=0)  # [N, hidden]
    all_labels = torch.cat(all_labels, dim=0).unsqueeze(1)  # [N, 1]

    df = pd.DataFrame(all_hidden.numpy())
    df['label'] = all_labels.numpy()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    del all_hidden, all_labels

class DebertaScratch(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.alpha = 0.2
        self.deberta = DebertaV2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, output_hidden_states=None):
        if output_hidden_states is None:
            output_hidden_states = labels is not None
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=output_hidden_states)
        cls_12 = outputs.last_hidden_state[:, 0]
        cls_12_drop = self.dropout(cls_12)
        logits = self.classifier(cls_12_drop)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            ce_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = ce_loss

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

def get_data(tokenizer):
    train = pd.read_csv("../corpus/imdb/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("../corpus/imdb/testData.tsv", header=0, delimiter="\t", quoting=3)
    train, val = train_test_split(train, test_size=0.2, random_state=42)
    def clean_review(review):
        review_text = BeautifulSoup(review, "html.parser").get_text()
        review_text = re.sub("[^a-zA-Z?!]"," ", review_text)
        return review_text
    
    train_dict = {'label': train["sentiment"], 'text': train['review'].apply(clean_review)}
    val_dict = {'label': val["sentiment"], 'text': val['review'].apply(clean_review)}
    test_dict = {"text": test['review'].apply(clean_review)}

    train_dataset = datasets.Dataset.from_dict(train_dict)
    val_dataset = datasets.Dataset.from_dict(val_dict)
    test_dataset = datasets.Dataset.from_dict(test_dict)

    def preprocess_function(examples):
        return tokenizer(
            examples['text'], 
            truncation=True,
            max_length=256)
    
    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)
    return tokenized_train, tokenized_val, tokenized_test, test


def predict_and_save_results(trainer, tokenized_test, batch_size=64, output_path="./predictions.csv"):
    model.eval()
    
    total_samples = len(tokenized_test)
    
    num_batches = (total_samples + batch_size - 1) // batch_size
    
    all_predictions = []
    
    for i in tqdm(range(num_batches), desc="Predicting"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_samples)
        
        batch_data = tokenized_test[start_idx:end_idx]
        prediction_outputs = trainer.predict(batch_data)
        preds = np.argmax(prediction_outputs[0][0], axis=-1)
        all_predictions.extend(preds)
        
    return all_predictions

if __name__ == '__main__':
    set_seed()
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_API_KEY"] = "30aeb9d20707f9f2e58fb4bdff330361304cdc45"

    tokenizer = DebertaV2Tokenizer.from_pretrained('../deberta')
    model = DebertaScratch.from_pretrained('../deberta')
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    tokenized_train, tokenized_val, tokenized_test, test = get_data(tokenizer)
    
    wandb.init(
    project="contrastic_learning", 
    name="deberta_v3_base_lora",  
    dir="./deberta_v3_base_lora"
    )

    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logists, labels = eval_pred
        preds = np.argmax(logists, axis=-1)
        return {
            "accuracy": acc_metric.compute(predictions=preds, references=labels)["accuracy"],
            "f1":       f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"],
        }

    
    training_args = TrainingArguments(
        output_dir="./weights",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        num_train_epochs=5,
        eval_strategy="no",
        save_strategy="no",
        logging_strategy="steps",
        data_seed=42,
        logging_steps=100,
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        optimizers=(optimizer, None),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    extract_and_save_hidden_states(
    model=model,
    dataloader=trainer.get_train_dataloader(),
    save_path='./results/lora_hidden2.csv',
    device=training_args.device
    )

    wandb.finish()
    prediction_outputs = trainer.predict(tokenized_test)
    test_pred = np.argmax(prediction_outputs[0], axis=-1).flatten()

    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv("./results/deberta_v3_base_lora.csv", index=False, quoting=3)