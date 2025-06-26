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

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        # [bsz, n_views, num_hiddens]
        # features = features.unsqueeze(1)
        features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask
        # compute log_prob
        
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1e-6)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class DebertaScratch(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.alpha = 0.2
        self.deberta = DebertaV2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        for name, param in self.deberta.named_parameters():
            if 'encoder.layer' in name:
                layer_num = int(name.split('.')[2])
                if layer_num < 8:
                    param.requires_grad = False

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
            cls_9 = outputs.hidden_states[8][:, 0]
            scl_fct = SupConLoss()
            features = torch.stack([cls_12, cls_9, cls_12_drop], dim=1)
            features = F.normalize(features, dim=-1)
            scl_loss = scl_fct(features, labels)

            loss = ce_loss + self.alpha * scl_loss

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
    optimizer = AdamW(get_parameters(model), weight_decay=0.01)
    tokenized_train, tokenized_val, tokenized_test, test = get_data(tokenizer)
    wandb.init(project="contrastic_learning", name='deberta_v3_base_scl_multiview')
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
    save_path='./results/cls_hidden2.csv',
    device=training_args.device
    )

    wandb.finish()
    prediction_outputs = trainer.predict(tokenized_test)
    test_pred = np.argmax(prediction_outputs[0], axis=-1).flatten()
    print(test_pred)

    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv("./results/deberta_multiview.csv", index=False, quoting=3)