from distill_config import (
    DistillationTrainingArguments,
    DistillationTrainer,
    AlphaScheduler
    )

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
    DebertaV2PreTrainedModel,
    TrainerCallback
    )
from peft import get_peft_model, LoraConfig
from utils import get_data
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from bs4 import BeautifulSoup
from tqdm import tqdm
from teacher_train import DebertaScratch

import re
import wandb
import evaluate
import random
import argparse
import swanlab

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of epochs to train")
    parser.add_argument("--dataset", type=str, default="sst2",
                        help="Dataset to use, SST2 or IMDB")
    parser.add_argument("--hp", type=str, default="cosine_descent",
                        help="The way alpha changes")
    parser.add_argument("--usetw", type=int, default=0,
                        help="Whether initialize with teacher model weights, 0 is not use, 1 is use")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # Best teacher
    teacher_id = {
        "sst2": "./model_checkpoints_sst2/checkpoint-500",
        "imdb": "./model_checkpoints_imdb/checkpoint-1100"
    }

    student_id = "./student_basemodel_sst2" if args.usetw == 1 else "./student_basemodel_zero" 
    
    set_seed()
    swanlab.login(api_key="3ImGDbNAVVcnvp5J3sagY", save=True)
    swanlab.init(
        project="Knowledge-Distillation",
        experiment_name=f"KD-{"Zero" if args.usetw == 0 else "TWeight"}-MultiLayers[1-6]-hp-{args.hp} training on {args.dataset}-epochs {args.epochs}",
        config={
            "model": student_id,
        }
    )
    
    tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-base')
    teacher = DebertaScratch.from_pretrained(teacher_id[(args.dataset).lower()])
    model = DebertaScratch.from_pretrained(student_id)

    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    tokenized_train, tokenized_test = get_data(tokenizer, args.dataset)

    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        # print("====================================")
        # print(eval_pred.__dict__.keys())
        
        logits, labels = eval_pred.predictions, eval_pred.label_ids
        # print(len(logits[0]))
        
        preds = np.argmax(logits[0], axis=-1)
        return {
            "accuracy": acc_metric.compute(predictions=preds, references=labels)["accuracy"],
            "f1":       f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"],
        }
    #gradient_accumulation_steps=4,
    training_args = DistillationTrainingArguments(
        output_dir=f"./KD_{"Zero" if args.usetw == 0 else "TWeight"}_MultiLayers[1-6]_checkpoints_{args.dataset}",
        per_device_train_batch_size=256,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=512,
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
        fp16=True,
        alpha=0.5,
        temperature=2.0,
        distill_layers=[1,2,3,4,5,6],
        hp_method=args.hp
    )

    total_steps = len(tokenized_train) // training_args.per_device_train_batch_size * training_args.num_train_epochs
    alpha_scheduler = AlphaScheduler(total_steps=total_steps, warmup_ratio=0.1)

    trainer = DistillationTrainer(
        model=model,
        teacher_model=teacher,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        optimizers=(optimizer, None),
        compute_metrics=compute_metrics,
        callbacks=[alpha_scheduler]
    )
    # trainer.evaluate()
    trainer.train()
    swanlab.finish()

