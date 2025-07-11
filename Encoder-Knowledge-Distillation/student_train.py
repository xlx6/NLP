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
from peft import (
    get_peft_model,
    LoraConfig
    )
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from bs4 import BeautifulSoup
from tqdm import tqdm
import re
import wandb
import evaluate
import random
import argparse
import swanlab


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


class DebertaScratch(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.deberta = DebertaV2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(self, input_ids=None, labels=None, attention_mask=None):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
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
            hidden_states=outputs.hidden_states
        )

def get_data(tokenizer, path):
    if path.lower() == 'sst2':
        target = 'sentence'
        dataset = datasets.load_dataset('glue', 'sst2')
        train_dataset, test_dataset = dataset['train'], dataset['validation']
    elif path.lower() == 'imdb':
        target = 'text'
        dataset = datasets.load_dataset('imdb')
        train_dataset, test_dataset = dataset['train'], dataset['test']
    else:
        raise ValueError('Dataset not supported')

    tag_re = re.compile(r'<.*?>')

    def preprocess_function(examples):
        return tokenizer(
            examples[target], 
            truncation=True,
            max_length=512)

    def clean_and_tokenize_batch(examples):
        cleaned = [tag_re.sub('', text) for text in examples[target]]
        return tokenizer(
            cleaned,
            truncation=True,
            max_length=512
        )

    tokenized_train = train_dataset.map(
        clean_and_tokenize_batch if target == 'text' else preprocess_function, 
        batched=True
    )
    tokenized_test = test_dataset.map(
        clean_and_tokenize_batch, 
        batched=True
    )

    return tokenized_train, tokenized_test

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of epochs to train")
    parser.add_argument("--dataset", type=str, default="sst2",
                        help="Dataset to use, SST2 or IMDB")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    set_seed()
    swanlab.login(api_key="3ImGDbNAVVcnvp5J3sagY", save=True)
    swanlab.init(
        project="Knowledge-Distillation",
        experiment_name=f"Student-Zero Model training on {args.dataset}-epochs {args.epochs}",
        config={
            "model": "deberta-v3-base",
        }
    )
    
    tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-base')
    model = DebertaScratch.from_pretrained('./student_basemodel_zero')
    # lora_config = get_lora_config()
    # model = get_peft_model(model, lora_config)
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    tokenized_train, tokenized_test = get_data(tokenizer, args.dataset)

    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        print(eval_pred)
        logits, labels = eval_pred.predictions, eval_pred.label_ids
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": acc_metric.compute(predictions=preds, references=labels)["accuracy"],
            "f1":       f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"],
        }
    #gradient_accumulation_steps=4,
    training_args = TrainingArguments(
        output_dir=f"./student_zero_model_checkpoints_{args.dataset}",
        per_device_train_batch_size=256,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=256,
        num_train_epochs=args.epochs,
        save_strategy="steps",
        eval_strategy="steps",
        logging_strategy="steps",
        data_seed=42,
        logging_steps=100,
        save_steps=100,
        eval_steps=100,
        save_total_limit=2,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        report_to="swanlab",
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        optimizers=(optimizer, None),
        compute_metrics=compute_metrics
    )
    # trainer.evaluate()
    trainer.train()
    swanlab.finish()
    
