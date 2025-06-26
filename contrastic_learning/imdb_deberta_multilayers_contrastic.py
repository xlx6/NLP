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
import swanlab
import wandb
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
os.environ["SWANLAB_PROJECT"]="contrastic_learning"

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
    def __init__(self, config, k=10, alpha=0.2, layers=[8, 11], level='sentence'):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.alpha = alpha
        self.k = k
        self.layers = layers
        self.level = level
        self.deberta = DebertaV2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.scl_fct = SupConLoss()
        self.loss_fct = nn.CrossEntropyLoss()
        self.post_init()

        print(self.layers)
        

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
            # [layers, bsz, n_hiddens]
            selected_embed = [outputs.hidden_states[i] for i in self.layers]
            cls_drops = [self.dropout(embed[:, 0, :]) for embed in selected_embed]

            ce_loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            scl_loss = 0
            for idx, layer in enumerate(self.layers):
                if self.level == 'token':
                    # [bsz, k]
                    top_k_indices = select_topk_tokens(outputs.attentions[layer], k=self.k)
                    selected_features = torch.gather(
                        selected_embed[idx],
                        dim=1,
                        index=top_k_indices.unsqueeze(-1).expand(-1, -1, selected_embed[idx].size(-1))
                    )
                elif self.level == 'sentence':
                    selected_features = torch.stack([selected_embed[idx][:, 0, :], cls_drops[idx]], dim=1)
                else:
                    raise ValueError(f"Invalid level: {self.level}")
                selected_features = F.normalize(selected_features, dim=-1)
                scl_loss += self.scl_fct(selected_features, labels)
            
            scl_loss /= len(self.layers)
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
    parser.add_argument("--layers", type=int, nargs='+', default=[8, 11], 
                       help="Layers to select for contrastive learning")
    parser.add_argument("--epochs", type=int, default=3, 
                       help="Number of epochs to train")
    parser.add_argument("--level", type=str, default='token', 
                       help="Level of contrastive learning, token or sentence")
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()
    
    set_seed()
    swanlab.login(api_key="3ImGDbNAVVcnvp5J3sagY", save=True)
    swanlab.init(
        project="contrastic_learning",
        config={
            "k": args.k,
            "alpha": args.alpha,
            "layers": args.layers,
            "epochs": args.epochs,
            "level": args.level
        }
    )
    
    tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-base')
    model = DebertaScratch.from_pretrained('microsoft/deberta-v3-base', k=args.k, alpha=args.alpha, layers=args.layers, level=args.level)
    model.alpha = args.alpha
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    tokenized_train, tokenized_val, tokenized_test, test = get_data(tokenizer)

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
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        eval_strategy="no",
        save_strategy="no",
        logging_strategy="steps",
        data_seed=42,
        logging_steps=100,
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="swanlab",
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
    layers = '-'.join(map(str, args.layers))
    extract_and_save_hidden_states(
    model=model,
    dataloader=trainer.get_eval_dataloader(),
    save_path=f'./results/{args.level}_level_hiddens_layers_{layers}.csv',
    device=training_args.device
    )

    
    prediction_outputs = trainer.predict(tokenized_test)

    swanlab.finish()
    test_pred = np.argmax(prediction_outputs[0], axis=-1).flatten()

    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv(f'./results/{args.level}_level_predict_layers_{layers}.csv', index=False, quoting=3)