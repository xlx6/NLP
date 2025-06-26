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
import argparse

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
from utils import (
    get_data,
    set_seed,
    extract_and_save_hidden_states,
    select_topk_tokens,
    SupConLoss
)

# os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

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
    def __init__(self, config, k=10, alpha=0.2):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.alpha = alpha
        self.k = k
        self.deberta = DebertaV2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.scl_fct = SupConLoss()
        self.loss_fct = nn.CrossEntropyLoss()
        self.post_init()
        

    def forward(
        self, 
        input_ids=None, 
        attention_mask=None, 
        labels=None, 
        output_hidden_states=None,
        output_attentions=None
        ):

        if output_hidden_states is None:
            output_hidden_states = labels is not None
        if output_attentions is None:
            output_attentions = labels is not None
        outputs = self.deberta(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions
            )
        cls_12 = outputs.last_hidden_state[:, 0]
        cls_12_drop = self.dropout(cls_12)
        logits = self.classifier(cls_12_drop)

        loss = None
        if labels is not None:
            ce_loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            top_k_indices = select_topk_tokens(outputs.attentions[-1], k=self.k)
            selected_features = torch.gather(
                outputs.last_hidden_state, 
                dim=1, 
                index=top_k_indices.unsqueeze(-1).expand(-1, -1, outputs.last_hidden_state.size(-1))
                )
            selected_features = F.normalize(selected_features, dim=-1)
            scl_loss = self.scl_fct(selected_features, labels)

            loss = ce_loss + self.alpha * scl_loss

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=100, 
                       help="Number of top-k tokens to select for contrastive learning")
    parser.add_argument("--alpha", type=float, default=0.2,
                       help="Weight for contrastive loss")
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()

    set_seed()
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_API_KEY"] = "30aeb9d20707f9f2e58fb4bdff330361304cdc45"

    tokenizer = DebertaV2Tokenizer.from_pretrained('../deberta')
    model = DebertaScratch.from_pretrained('../deberta', k=args.k, alpha=args.alpha)
    model.alpha = args.alpha
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    tokenized_train, tokenized_val, tokenized_test, test = get_data(tokenizer)
    
    wandb.init(
    project="contrastic_learning", 
    name=f"token_level_lora_k{args.k}_alpha{args.alpha}",  
    dir="./deberta_v3_token_level_scl_lora",
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
        num_train_epochs=3,
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
    save_path=f'./results/token_level_hiddens_k{args.k}_alpha{args.alpha}.csv',
    device=training_args.device
    )

    wandb.finish()
    prediction_outputs = trainer.predict(tokenized_test)
    test_pred = np.argmax(prediction_outputs[0], axis=-1).flatten()

    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv(f"./results/token_level_k{args.k}_alpha{args.alpha}.csv", index=False, quoting=3)