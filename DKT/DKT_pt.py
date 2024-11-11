import math
import numpy as np
import os
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, accuracy_score
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class DKTModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_rate):
        super(DKTModel, self).__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dr = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.emb(x)
        output, _ = self.lstm(x)
        output = self.dr(output)
        output = self.fc(output)
        output = self.sigmoid(output)
        return output

class DKTDataset(Dataset):
    def __init__(self, data, vocab_to_idx, max_seq_len):
        self.data = data
        self.vocab_to_idx = vocab_to_idx
        self.max_seq_len = max_seq_len
        self.samples = []
        self._prep_samples()

    def _prep_samples(self):
        user_groups = self.data.groupby('user_xid')
        for uid, u_data in user_groups:
            u_data = u_data.sort_values('start_time')
            u_data['skill_id'] = u_data['skill_id'].astype(str)
            u_data['skill_id_x_corr'] = u_data['skill_id']
            u_data.loc[u_data['discrete_score'] == 0, 'skill_id_x_corr'] += '+0'
            u_data.loc[u_data['discrete_score'] == 1, 'skill_id_x_corr'] += '+1'
            encoded_seq = [self.vocab_to_idx.get(s, self.vocab_to_idx['<UNK>']) for s in u_data['skill_id_x_corr']]
            correct_seq = u_data['discrete_score'].to_numpy()

            seq_len = len(encoded_seq)
            for start_idx in range(0, seq_len, self.max_seq_len):
                end_idx = min(start_idx + self.max_seq_len, seq_len)
                sub_feat_seq = encoded_seq[start_idx:end_idx]
                sub_correct_seq = correct_seq[start_idx:end_idx]
                # Pad the sequence
                pad_len = self.max_seq_len - len(sub_feat_seq)
                input_seq = sub_feat_seq + [self.vocab_to_idx['<PAD>']] * pad_len

                label_seq = np.full((self.max_seq_len, len(self.vocab_to_idx)), -1, dtype=np.float32)
                for i, (enc, corr) in enumerate(zip(sub_feat_seq, sub_correct_seq)):
                    # Ensure corr is 0 or 1
                    corr = int(corr)
                    if corr not in [0, 1]:
                        corr = 0  # Handle invalid values
                    label_seq[i, enc] = corr
                self.samples.append((input_seq, label_seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_seq, label = self.samples[idx]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(label, dtype=torch.float32)

def compute_auc(y_true, y_pred):
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    try:
        auc = roc_auc_score(y_true_np, y_pred_np)
    except ValueError:
        auc = 0.0
    return auc

def compute_accuracy(y_true, y_pred):
    y_pred_labels = (y_pred.cpu().numpy() >= 0.5).astype(int)
    y_true_labels = y_true.cpu().numpy().astype(int)
    acc = accuracy_score(y_true_labels, y_pred_labels)
    return acc

class DKT:
    def __init__(self, batch_size=64, num_steps=50, hidden_dim_size=124, dropout_rate=0.2, learning_rate=1e-3, verbose=True, gpu_num=1):
        self.model = None
        self.vocab = []
        self.vocab_size = 0
        self.num_steps = num_steps
        self.verbose = verbose
        self.hidden_dim_size = hidden_dim_size
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.vocab_to_idx = None
        self.idx_to_vocab = None
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def preprocess(self, data, fitting=True):
        data = data.copy()
        data['start_time'] = pd.to_datetime(
            data['start_time'],
            infer_datetime_format=True,
            utc=True,
            errors='coerce'  # Handle parsing errors
        )
        data = data[['user_xid', 'skill_id', 'discrete_score', 'start_time']].sort_values(by=['user_xid', 'start_time'])
        data = data.fillna({'skill_id': 'unknown_skill', 'discrete_score': 0})
        data['skill_id'] = data['skill_id'].astype(str)
        data['discrete_score'] = pd.to_numeric(data['discrete_score'], errors='coerce').fillna(0).astype(int)
        data['discrete_score'] = data['discrete_score'].clip(lower=0, upper=1)  # Ensure values are 0 or 1

        if fitting:
            un = data['skill_id'].unique()
            zer = un + '+0'
            on = un + '+1'

            self.vocab = np.concatenate([zer, on])
            self.vocab_size = len(self.vocab) + 2
            self.embedding_dim = max(50, math.ceil(math.log(self.vocab_size)))
            self.vocab_to_idx = {token: idx + 2 for idx, token in enumerate(self.vocab)}
            self.vocab_to_idx['<PAD>'] = 0
            self.vocab_to_idx['<UNK>'] = 1  # Corrected key
            self.idx_to_vocab = {idx: token for token, idx in self.vocab_to_idx.items()}
        return data

    def train(self, train_data, num_epochs=5):
        train_data = self.preprocess(train_data, fitting=True)
        dataset = DKTDataset(train_data, self.vocab_to_idx, self.num_steps)
        if len(dataset) == 0:
            print("No data to train on. Exiting training.")
            return
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = DKTModel(self.vocab_size, self.embedding_dim, self.hidden_dim_size, self.dropout_rate).to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            total_preds = []
            total_labels = []

            for batch_idx, (input_seq, label_seq) in enumerate(
                    tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', disable=not self.verbose)):
                input_seq = input_seq.to(self.device)
                label_seq = label_seq.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_seq)

                # Mask out padding and entries with -1 labels
                mask = (label_seq != -1)
                y_pred = outputs[mask]
                y_true = label_seq[mask]

                if y_true.numel() == 0:
                    continue  # Skip if no valid entries in batch

                # Ensure y_true is within [0, 1]
                y_true = torch.clamp(y_true, min=0.0, max=1.0)

                loss = criterion(y_pred, y_true)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                total_preds.append(y_pred.detach())
                total_labels.append(y_true.detach())

            avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else float('inf')

            if total_preds:
                all_preds = torch.cat(total_preds)
                all_labels = torch.cat(total_labels)
                auc = compute_auc(all_labels, all_preds)
                acc = compute_accuracy(all_labels, all_preds)
            else:
                auc = 0.0
                acc = 0.0

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, AUC: {auc:.4f}, Accuracy: {acc:.4f}")

    def eval(self, val_data):
        if self.model is None:
            print('Model has not been trained yet')
            return

        val_data = self.preprocess(val_data, fitting=False)
        dataset = DKTDataset(val_data, self.vocab_to_idx, self.num_steps)
        if len(dataset) == 0:
            print("No data to evaluate on.")
            return
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        total_loss = 0.0
        total_preds = []
        total_labels = []
        criterion = nn.BCELoss()

        with torch.no_grad():
            for batch_idx, (input_seq, label_seq) in enumerate(
                    tqdm(dataloader, desc="Evaluation", disable=not self.verbose)):
                input_seq = input_seq.to(self.device)
                label_seq = label_seq.to(self.device)

                outputs = self.model(input_seq)

                # Mask out padding and entries with -1 labels
                mask = (label_seq != -1)
                y_pred = outputs[mask]
                y_true = label_seq[mask]

                if y_true.numel() == 0:
                    continue  # Skip if no valid entries in batch

                # Ensure y_true is within [0, 1]
                y_true = torch.clamp(y_true, min=0.0, max=1.0)

                loss = criterion(y_pred, y_true)
                total_loss += loss.item()

                total_preds.append(y_pred)
                total_labels.append(y_true)

        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else float('inf')
        if total_preds:
            all_preds = torch.cat(total_preds)
            all_labels = torch.cat(total_labels)
            auc = compute_auc(all_labels, all_preds)
            acc = compute_accuracy(all_labels, all_preds)
        else:
            auc = 0.0
            acc = 0.0

        print(f"Evaluation Results - Loss: {avg_loss:.4f}, AUC: {auc:.4f}, Accuracy: {acc:.4f}")

        return auc
