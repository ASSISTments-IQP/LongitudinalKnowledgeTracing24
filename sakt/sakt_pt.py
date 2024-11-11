import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

class SAKTDataset(Dataset):
    def __init__(self, df: pd.DataFrame, exercise_map: dict, num_steps: int):
        self.df = df.copy()
        self.df['start_time'] = pd.to_datetime(self.df['start_time'], utc=True)
        self.df = self.df[['user_xid', 'old_problem_id', 'discrete_score', 'start_time']].sort_values(
            by=['user_xid', 'start_time'])
        self.exercise_map = exercise_map
        self.num_steps = num_steps
        self.samples = []

        self._prepare_samples()

    def _prepare_samples(self):
        user_xids = self.df['user_xid'].unique()

        for user_xid in user_xids:
            user_data = self.df[self.df['user_xid'] == user_xid]
            exercise_seq = [self.exercise_map.get(id, 0) for id in user_data['old_problem_id']]
            response_seq = user_data['discrete_score'].astype(int).tolist()

            for idx in range(1, len(exercise_seq)):
                start_idx = max(0, idx - self.num_steps)
                past_exercises = [0] * (self.num_steps - (idx - start_idx)) + exercise_seq[start_idx:idx]
                past_responses = [0] * (self.num_steps - (idx - start_idx)) + response_seq[start_idx:idx]
                current_exercise = exercise_seq[idx]
                target_response = response_seq[idx]

                self.samples.append((past_exercises, past_responses, current_exercise, target_response))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        past_exercises, past_responses, current_exercise, target_response = self.samples[idx]
        return (
            torch.tensor(past_exercises, dtype=torch.long),
            torch.tensor(past_responses, dtype=torch.long),
            torch.tensor(current_exercise, dtype=torch.long),
            torch.tensor(target_response, dtype=torch.float32)
        )


class SAKTModel(nn.Module):
    def __init__(self, num_steps=50, d_model=128, num_heads=8, dropout_rate=0.2, num_exercises=1):
        super(SAKTModel, self).__init__()
        self.num_steps = num_steps
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.exercise_embedding = nn.Embedding(num_exercises + 1, d_model)
        self.response_embedding = nn.Embedding(2, d_model)
        self.position_embedding = nn.Embedding(num_steps, d_model)

        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout_rate)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, past_exercises, past_responses, current_exercises):
        past_exercise_emb = self.exercise_embedding(past_exercises)
        past_response_emb = self.response_embedding(past_responses)
        interactions_emb = past_exercise_emb + past_response_emb
        seq_len = past_exercises.size(1)
        position_ids = torch.arange(seq_len, device=past_exercises.device).unsqueeze(0)
        position_emb = self.position_embedding(position_ids)
        interactions_emb += position_emb
        attention_mask = (past_exercises == 0)
        query = self.exercise_embedding(current_exercises).unsqueeze(1).permute(1, 0, 2)
        key = value = interactions_emb.permute(1, 0, 2)
        attn_out, _ = self.attention(query, key, value, key_padding_mask=attention_mask)
        attn_out = attn_out.permute(1, 0, 2)
        out1 = self.norm1(attn_out + query.permute(1, 0, 2))
        ffn_out = self.ffn(out1)
        out2 = self.norm2(ffn_out + out1)
        output = self.output(out2).squeeze(-1)
        return output

    def compute_loss(self, predictions, targets):
        return self.loss_fn(predictions, targets)


def preprocess_data(df: pd.DataFrame):
    df = df.dropna()
    unique_problems = df['old_problem_id'].unique()

    exercise_map = {problem: idx for idx, problem in enumerate(unique_problems, start=1)}
    return exercise_map, len(exercise_map)


import numpy as np

def train(model, train_loader, val_loader=None, num_epochs=5, clip_value=1.0, lr=1e-3, patience=1):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
    best_val_auc = 0.0
    epochs_since_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        train_losses, all_labels, all_preds = [], [], []

        for past_exercises, past_responses, current_exercises, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            preds = model(past_exercises.cuda(), past_responses.cuda(), current_exercises.cuda())
            preds = preds.squeeze(-1)
            loss = model.compute_loss(preds, targets.cuda())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            train_losses.append(loss.item())
            all_labels.extend(targets.cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        valid_indices = ~np.isnan(all_preds)
        all_labels = all_labels[valid_indices]
        all_preds = all_preds[valid_indices]

        if len(all_labels) > 0 and len(all_preds) > 0:
            epoch_auc = roc_auc_score(all_labels, all_preds)
        else:
            epoch_auc = 0.0
            print("Warning: No valid predictions for AUC calculation.")

        print(f"Epoch {epoch + 1} - Loss: {np.mean(train_losses):.4f}, AUC: {epoch_auc:.4f}")

        if val_loader is not None:
            val_auc, val_loss = evaluate(model, val_loader)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                epochs_since_improvement = 0
                torch.save(model.state_dict(), 'best_model_weights.pth')
            else:
                epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                print("Early stopping triggered.")
                break

        scheduler.step()


def evaluate(model, val_loader):
    model.eval()
    val_losses, all_labels, all_preds = [], [], []

    with torch.no_grad():
        for past_exercises, past_responses, current_exercises, targets in val_loader:
            preds = model(past_exercises.cuda(), past_responses.cuda(), current_exercises.cuda())
            preds = preds.squeeze(-1)
            loss = model.compute_loss(preds, targets.cuda())
            val_losses.append(loss.item())
            all_labels.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    valid_indices = ~np.isnan(all_preds)
    all_labels = all_labels[valid_indices]
    all_preds = all_preds[valid_indices]
    if len(all_labels) > 0 and len(all_preds) > 0:
        val_auc = roc_auc_score(all_labels, all_preds)
    else:
        val_auc = 0.0
        print("Warning: No valid predictions for AUC calculation.")
    
    val_loss = np.mean(val_losses)
    return val_auc, val_loss
