import math
import numpy as np
import os
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, accuracy_score
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional

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
        return self.sigmoid(output)
class DKTDataset(Dataset):
    def __init__(self, data: pd.DataFrame, vocab_to_idx: Dict[str, int], max_seq_len: int, feature_col: str) -> None:
        self.data: pd.DataFrame = data
        self.feature_col: str = feature_col
        self.vocab_to_idx = vocab_to_idx
        self.max_seq_len: int = max_seq_len
        self.samples: List[Tuple[List[int]]] = []
        self._prep_samples()

    def _prep_samples(self) -> None:
        user_groups = self.data.groupby('user_xid')
        
        print(f' user_groups: \n {user_groups}')
        for uid, u_data in user_groups:
            
            # print(f'uid iterator: {uid}')
            #
            # print(f'u_data df: \n{u_data}')
            
            u_data = u_data.sort_values('start_time')
            # print(f'u_data sorted by start time: \n{u_data}')
            
            u_data[self.feature_col] = u_data[self.feature_col].astype(str)
            # print(f' u_dataw with relevant feature col cast to str:  \n {u_data}')
            
            x_cor_col_name:str = f'{self.feature_col}_x_corr'
            # print(f'x_cor_col_name: {x_cor_col_name}')
            
            u_data[x_cor_col_name] = u_data[self.feature_col]
            # print('u_data before +0s {u_data}')
            
            u_data.loc[u_data['discrete_score'] == 0, x_cor_col_name] += '+0'
            # print(f'u_data after +0 but before +1s: {u_data}')
            
            u_data.loc[u_data['discrete_score'] == 1, x_cor_col_name] += '+1'
            # print(f'u_data after +0 and + 1: {u_data}')

            encoded_seq: List[int] = [
                self.vocab_to_idx.get(s, self.vocab_to_idx['<UNK>']) 
                for s in u_data[x_cor_col_name]
            ]
            # print(f'encoded_seq: {encoded_seq}')
            
            correct_seq: np.ndarray = u_data['discrete_score'].to_numpy()
            # print(f' correct_seq: {correct_seq}')

            seq_len:int = len(encoded_seq)
            # print(f'sequence length: {seq_len}')
            
            for start_idx in range(0, seq_len, self.max_seq_len):

                # print(f'start_idx: {start_idx}')
                
                end_idx:int = min(start_idx + self.max_seq_len, seq_len)
                # print(f'end_idx: {end_idx}')
                
                sub_feat_seq:List[int] = encoded_seq[start_idx:end_idx]
                # print(f'sub_feat_seq: {sub_feat_seq}')
                
                sub_correct_seq: np.ndarray = correct_seq[start_idx:end_idx]
                # print(f'sub_correct_seq: {sub_correct_seq}')
                
                pad_len: int = self.max_seq_len - len(sub_feat_seq)
                # print(f'len of paddign: {pad_len}')
                
                input_seq: List[int] = sub_feat_seq + [self.vocab_to_idx['<PAD>']] * pad_len
                # print(f'input_sequence: {input_seq}')
                
                label_seq: np.ndarray= np.full((self.max_seq_len, len(self.vocab_to_idx)), -1, dtype=np.float32)
                # print(f'label_seq: {label_seq}')
                
                for i, (enc, corr) in enumerate(zip(sub_feat_seq, sub_correct_seq)):
                    # print(f'i: {i}')
                    # print(f'enc: {enc}')
                    # print(f'corr: {corr}')
                    if enc == 0:
                        continue
                    label_seq[i, enc] = corr
                    # print(f'label_seq: {label_seq}')
                
                self.samples.append((input_seq, label_seq))
                # print(f'self.samples: {self.samples}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx:int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_seq, label = self.samples[idx]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(label, dtype=torch.float32)

def compute_auc(y_true: torch.Tensor, y_pred: torch.Tensor):
    y_true_np: np.ndarray = y_true.cpu().numpy()
    y_pred_np: np.ndarray = y_pred.cpu().numpy()

    # Debugging AUC inputs
    # print(f"[DEBUG] y_true_np[:10]: {y_true_np[:10]}")
    # print(f"[DEBUG] y_pred_np[:10]: {y_pred_np[:10]}")
    # print(f"[DEBUG] Label distribution: {np.bincount(y_true_np.astype(int))}")
    # print(f"[DEBUG] Prediction variance: {np.var(y_pred_np):.4f}")

    try:
        auc = roc_auc_score(y_true_np, y_pred_np)
    except ValueError:
        auc = 0.0

    # Debug random baseline AUC for comparison
    random_preds = np.random.uniform(size=y_true_np.shape)
    random_auc = roc_auc_score(y_true_np, random_preds)
    # print(f"[DEBUG] Random baseline AUC: {random_auc:.4f}")

    return auc


def compute_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    y_pred_labels: np.ndarray = y_pred.cpu().numpy().astype(int)
    y_true_labels: np.ndarray = y_true.cpu().numpy().astype(int)
    acc: float = accuracy_score(y_true_labels, y_pred_labels)
    return acc


class DKT:
    def __init__(self, batch_size:int=64, num_steps:int=10, hidden_dim_size:int=124, dropout_rate:float=0.2, learning_rate:float=1e-3, regularization_lambda=1e-3, verbose:bool=True, gpu_num:int=1, feature_col:str = 'skill_id')-> None:
        self.model:Optional[nn.Module] = None
        self.feature_col: str = feature_col
        self.vocab: List[str] = []
        self.vocab_size: int = 0
        self.num_steps: int = num_steps
        self.verbose: bool = verbose
        self.hidden_dim_size: int = hidden_dim_size
        self.batch_size: int= batch_size
        self.dropout_rate: float = dropout_rate
        self.learning_rate: float = learning_rate
        self.reg_lambda = regularization_lambda
        self.vocab_to_idx: Optional[Dict[str, int]] = None
        self.idx_to_vocab: Optional[Dict[str, int]] = None
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def preprocess(self, data: pd.DataFrame, fitting=True)-> pd.DataFrame:
        data = data.copy()

        data['start_time'] = pd.to_datetime(data['start_time'], utc=True, errors='coerce')
        # print(f'df after converting start time to a pandas datetime obj: \n {data}')

        data = data[['user_xid', self.feature_col, 'discrete_score', 'start_time']].sort_values(by=['user_xid', 'start_time'])
        # print(f'df sorted by userid and start time: \n {data}')

        data[self.feature_col] = data[self.feature_col].astype(str)
        # print(f' df after converting feat cols to str: \n {data}')

        if fitting:
            un: np.ndarray = data[self.feature_col].unique()
            # print(f'un: {un}')

            zer: np.ndarray = un + '+0'
            # print(f'zer: {zer}')

            on: np.ndarray = un + '+1'
            # print(f'on: {on}')

            self.vocab = np.concatenate([zer, on])
            # print(f'vocab: {self.vocab}')

            self.vocab_size = len(self.vocab) + 2
            # print(f'self.vocab_size: {self.vocab_size}')
            
            self.embedding_dim = max(10, math.ceil(math.log(self.vocab_size)))
            # print(f'self.embedding_dim: {self.embedding_dim}')

            self.vocab_to_idx = {token: idx + 2 for idx, token in enumerate(self.vocab)}
            # print(f'self.vocab_to_idx: {self.vocab_to_idx}')

            self.vocab_to_idx['<PAD>'] = 0
            # print(f'self.vocab_to_idx: {self.vocab_to_idx}')

            self.vocab_to_idx['<UNK>'] = 1
            # print(f'self.vocab_to_idx: {self.vocab_to_idx}')

            self.idx_to_vocab = {idx: token for token, idx in self.vocab_to_idx.items()}
            # print(f'self.idx_to_vocab: {self.idx_to_vocab}')

        return data

    def fit(self, train_data: pd.DataFrame, num_epochs: int=5) -> None:
        train_data = self.preprocess(train_data, fitting=True)
        dataset = DKTDataset(train_data, self.vocab_to_idx, self.num_steps, self.feature_col)
        if len(dataset) == 0:
            print("No data to train on. Exiting training.")
            return
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = DKTModel(self.vocab_size, self.embedding_dim, self.hidden_dim_size, self.dropout_rate).to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.reg_lambda)

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            total_preds = []
            total_labels = []

            for batch_idx, (input_seq, label_seq) in enumerate(tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', disable=not self.verbose)):
                input_seq = input_seq.to(self.device)
                label_seq = label_seq.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_seq)

                mask = (label_seq != -1) & (input_seq.unsqueeze(-1) != self.vocab_to_idx['<PAD>'])
                y_pred = outputs[mask]
                # print(y_pred)
                y_true = label_seq[mask]
                # print(y_true)

                if y_true.numel() == 0:
                    continue

                y_true = torch.clamp(y_true, min=0.0, max=1.0)

                loss = criterion(y_pred, y_true)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_preds.append(y_pred.detach())
                total_labels.append(y_true.detach())

                
                
                original_keys = [self.idx_to_vocab.get(idx.item(), '<UNK>') for idx in input_seq[0]]
                # print("[DEBUG] Original Keys:", original_keys)
                # print("[DEBUG] Sample predictions:")
                # print(f"Predictions: {y_pred.detach().cpu().numpy()}")
                # print(f"Labels: {y_true.detach().cpu().numpy()}")

            avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else float('inf')
            print(f"[DEBUG] Epoch {epoch + 1}, Average Loss: {avg_loss}")

            if total_preds:
                all_preds = torch.cat(total_preds)
                all_labels = torch.cat(total_labels)
                auc = compute_auc(all_labels, all_preds)
                acc = compute_accuracy(all_labels, all_preds)
            else:
                auc = 0.0
                acc = 0.0

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, AUC: {auc:.4f}, Accuracy: {acc:.4f}")

    def evaluate(self, val_data):
        if self.model is None:
            print('Model has not been trained yet')
            return

        val_data = self.preprocess(val_data, fitting=False)
        dataset = DKTDataset(val_data, self.vocab_to_idx, self.num_steps, self.feature_col)
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
            for batch_idx, (input_seq, label_seq) in enumerate(tqdm(dataloader, desc="Evaluation", disable=not self.verbose)):
                input_seq = input_seq.to(self.device)
                label_seq = label_seq.to(self.device)

                outputs = self.model(input_seq)

                mask = (label_seq != -1) & (input_seq.unsqueeze(-1) != self.vocab_to_idx['<PAD>'])
                y_pred = outputs[mask]
                y_true = label_seq[mask]

                if y_true.numel() == 0:
                    continue

                y_true = torch.clamp(y_true, min=0.0, max=1.0)

                loss = criterion(y_pred, y_true)
                total_loss += loss.item()

                total_preds.append(y_pred)
                total_labels.append(y_true)

                original_keys = [self.idx_to_vocab.get(idx.item(), '<UNK>') for idx in input_seq[0]]
                # print("[DEBUG] Original Keys:", original_keys)
                # print("[DEBUG] Sample predictions:")
                # print(f"Predictions: {y_pred.detach().cpu().numpy()}")
                # print(f"Labels: {y_true.detach().cpu().numpy()}")

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

        return auc, acc
