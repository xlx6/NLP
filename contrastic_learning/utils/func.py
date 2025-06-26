import random
import os
import re
import random
import numpy as np
import pandas as pd
import datasets
import torch
import torch.nn as nn
from tqdm import tqdm
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cluster_views(features: torch.Tensor, n_clusters: int) -> torch.Tensor:
    """
    对每个样本的多个视角特征进行聚类，并返回聚类中心特征。

    input:
        features:  [bsz, n_views, n_hiddens]
        n_clusters: number of cluster

    return:
        clustered_features: [bsz, n_clusters, n_hiddens]
    """
    bsz, n_views, n_hiddens = features.shape
    clustered_features = []

    for i in range(bsz):
        # [n_views, n_hiddens]
        views = features[i]  

        views_np = views.detach().cpu().numpy()

        # cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
        kmeans.fit(views_np)
        centers_np = kmeans.cluster_centers_

        centers = torch.tensor(centers_np, dtype=features.dtype, device=features.device)
        clustered_features.append(centers)

    # [bsz, n_clusters, n_hiddens]
    return torch.stack(clustered_features, dim=0)


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

def get_data(tokenizer, train_path='../corpus/imdb/labeledTrainData.tsv', test_path='../corpus/imdb/testData.tsv'):
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
            max_length=510)
    
    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)
    return tokenized_train, tokenized_val, tokenized_test, test

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

def select_topk_tokens(attention_scores, k):
    """
    选择`[CLS]`关注度最高的k个tokens
    input: 
        attention_scores:  [bsz, n_head, sq_len, sq_len]
        k: num select
    return:
        top_k_tokens: [bsz, k], index of selected
    """
    # [batch_size, num_heads, seq_len, seq_len]
    batch_size, num_heads, seq_len, _ = attention_scores.shape
    
    # [batch_size, num_heads, seq_len]
    cls_attention_scores = attention_scores[:, :, 0, :]  
    
    # [batch_size, seq_len]
    attention_scores_avg = cls_attention_scores.mean(dim=1)  
    attention_scores_avg[:, 0] = -1e9
    
    _, top_k_indices = torch.topk(attention_scores_avg, k, dim=-1, largest=True, sorted=False)
    
    # [bsz, k]
    return top_k_indices
