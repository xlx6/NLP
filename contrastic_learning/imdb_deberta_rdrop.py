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
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from bs4 import BeautifulSoup
import re
import wandb
import evaluate
import random

os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def KL(input, target, reduction="mean"):
    input = input.float()
    target = target.float()
    loss = F.kl_div(
        F.log_softmax(input, dim=-1, dtype=torch.float32),
        F.softmax(target, dim=-1, dtype=torch.float32), 
        reduction=reduction
        )
    return loss

def get_parameters(model, model_init_lr=2e-5, scaler_factor=0.95, classifier_lr=1e-4):
    parameters = []
    for i in range(8, 12):
        layer_params = {
            "params": [],
            "lr": model_init_lr * (scaler_factor ** (11 - i))
        }
        for name, param in model.named_parameters():
            if f'encoder.layer.{i}.' in name and param.requires_grad:
                layer_params["params"].append(param)
        parameters.append(layer_params)
    
    classifier_params = {
        "params": [],
        'lr': 1e-4
    }
    for name, param in model.named_parameters():
        if 'classifier' in name and param.requires_grad:
            classifier_params["params"].append(param)
    parameters.append(classifier_params)
    return parameters

class DebertaScratch(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.deberta = DebertaV2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        for name, param in self.deberta.named_parameters():
            if 'encoder.layer' in name:
                layer_num = int(name.split('.')[2])
                if layer_num < 8:
                    param.requires_grad = False

        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss_fct = nn.CrossEntropyLoss()
        loss = None
        if labels is not None:
            main_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            kl_outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
            kl_output = kl_outputs.last_hidden_state[:, 0]
            kl_output = self.dropout(kl_output)
            kl_logits = self.classifier(kl_output)
            kl_loss = (KL(logits, kl_logits) + KL(kl_logits, logits)) / 2.
            ce_loss = loss_fct(kl_logits.view(-1, self.num_labels), labels.view(-1))
            loss = main_loss + ce_loss + 0.2 * kl_loss

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

def get_data(tokenizer):
    train = pd.read_csv("/kaggle/input/kumarmanoj-bag-of-words-meets-bags-of-popcorn/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("/kaggle/input/kumarmanoj-bag-of-words-meets-bags-of-popcorn/testData.tsv", header=0, delimiter="\t", quoting=3)
    train, val = train_test_split(train, test_size=0.2, random_state=42)
    test = test
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

if __name__ == '__main__':
    set_seed()
    os.environ["WANDB_API_KEY"] = "30aeb9d20707f9f2e58fb4bdff330361304cdc45"
    tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-base')
    model = DebertaScratch.from_pretrained('microsoft/deberta-v3-base')
    optimizer = AdamW(get_parameters(model), weight_decay=0.01)
    tokenized_train, tokenized_val, tokenized_test, test = get_data(tokenizer)
    # wandb.init(project="contrastic_learning", name='deberta_v3_base_rdrop_tempreture0.2')
    # wandb.watch(model, log="all", log_freq=50)
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
        output_dir="./results",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        data_seed=42,
        logging_steps=100,
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="wandb",
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

    # trainer.train()
    # trainer.evaluate()
    # wandb.finish()

    prediction_outputs = trainer.predict(tokenized_test)
    test_pred = np.argmax(prediction_outputs[0], axis=-1).flatten()
    print(test_pred)

    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv("./deberta_v3_base_rdrop.csv", index=False, quoting=3)