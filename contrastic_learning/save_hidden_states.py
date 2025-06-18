import torch
import pandas as pd
from tqdm import tqdm
import os

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
            cls_output = outputs.last_hidden_state[:, 0]

            all_hidden.append(cls_output.detach().cpu())
            all_labels.append(labels.detach().cpu())

    all_hidden = torch.cat(all_hidden, dim=0)  # [N, hidden]
    all_labels = torch.cat(all_labels, dim=0).unsqueeze(1)  # [N, 1]

    df = pd.DataFrame(all_hidden.numpy())
    df['label'] = all_labels.numpy()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
