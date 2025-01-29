import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os, gc
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

class SAKTDataset(Dataset):
    def __init__(self, df: pd.DataFrame, exercise_map: dict, num_steps: int, feature_col='skill_id'):
        self.feature_col = feature_col
        self.df = df.copy()
        self.df['start_time'] = pd.to_datetime(self.df['start_time'], utc=True)
        self.df = self.df[['user_xid', self.feature_col, 'discrete_score', 'start_time']].sort_values(
            by=['user_xid', 'start_time'])
        self.exercise_map = exercise_map
        self.num_steps = num_steps
        self.samples = []

        self._prepare_samples()

    def _prepare_samples(self):
        user_xids = self.df['user_xid'].unique()

        for user_xid in user_xids:
            user_data = self.df[self.df['user_xid'] == user_xid]
            exercise_seq = [self.exercise_map.get(id, 0) for id in user_data[self.feature_col]]
            response_seq = user_data['discrete_score'].astype(int).tolist()

            for idx in range(1, len(exercise_seq)):
                start_idx = max(0, idx - self.num_steps)
                seq_length = idx - start_idx
                pad_length = self.num_steps - seq_length

                past_exercises = [0] * pad_length + exercise_seq[start_idx:idx]
                past_responses = [0] * pad_length + response_seq[start_idx:idx]
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
    def __init__(self, num_steps=50, batch_size = 32, d_model=128, num_heads=8, dropout_rate=0.2, init_learning_rate=1e-3, learning_decay_rate=0.98, feature_col='skill_id', gpu_num=0):
        super(SAKTModel, self).__init__()
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.init_learning_rate = init_learning_rate
        self.learning_decay_rate = learning_decay_rate
        self.feature_col = feature_col

        # Placeholders for embeddings; will be initialized during fit
        self.exercise_embedding = None
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
        self.output_layer = nn.Linear(d_model, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()

        # These attributes will be set during training
        self.exercise_map = None
        self.num_exercises = None

        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def preprocess_data(self, df: pd.DataFrame):
        """
        Preprocesses the data and creates the exercise_map.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing interaction data.

        Returns:
            df (pd.DataFrame): Cleaned DataFrame.
            exercise_map (dict): Mapping from feature IDs to indices.
            num_exercises (int): Total number of unique exercises.
        """
        df = df.dropna()
        unique_features = df[self.feature_col].unique()
        exercise_map = {feature: idx for idx, feature in enumerate(unique_features, start=1)}
        num_exercises = len(exercise_map)
        return df, exercise_map, num_exercises

    def forward(self, past_exercises, past_responses, current_exercises):
        past_exercise_emb = self.exercise_embedding(past_exercises)  # Shape: (batch_size, seq_len, d_model)
        past_response_emb = self.response_embedding(past_responses)  # Shape: (batch_size, seq_len, d_model)
        interactions_emb = past_exercise_emb + past_response_emb     # Shape: (batch_size, seq_len, d_model)

        seq_len = past_exercises.size(1)
        position_ids = torch.arange(seq_len, device=past_exercises.device).unsqueeze(0)  # Shape: (1, seq_len)
        position_emb = self.position_embedding(position_ids)                             # Shape: (1, seq_len, d_model)
        interactions_emb += position_emb                                                 # Shape: (batch_size, seq_len, d_model)

        attention_mask = (past_exercises == float('-inf'))  # Shape: (batch_size, seq_len)

        query = self.exercise_embedding(current_exercises).unsqueeze(1).permute(1, 0, 2)  # Shape: (1, batch_size, d_model)
        key = value = interactions_emb.permute(1, 0, 2)                                   # Shape: (seq_len, batch_size, d_model)

        attn_output, _ = self.attention(query, key, value, key_padding_mask=attention_mask)
        attn_output = attn_output.permute(1, 0, 2)  # Shape: (batch_size, 1, d_model)
        out1 = self.norm1(attn_output + query.permute(1, 0, 2))  # Shape: (batch_size, 1, d_model)
        ffn_out = self.ffn(out1)                                 # Shape: (batch_size, 1, d_model)
        out2 = self.norm2(ffn_out + out1)                        # Shape: (batch_size, 1, d_model)
        output = self.output_layer(out2).squeeze(-1).squeeze(-1) # Shape: (batch_size)
        return output

    def compute_loss(self, predictions, targets):
        return self.loss_fn(predictions, targets)

    def fit(self, df: pd.DataFrame,  num_epochs=5, patience=5, validation_split=0.1):
        """
        Trains the model using the provided DataFrame.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing interaction data.
            batch_size (int): Batch size for training.
            num_epochs (int): Number of training epochs.
            lr (float): Learning rate for the optimizer.
            patience (int): Number of epochs to wait for improvement before early stopping.
            validation_split (float): Proportion of data to use for validation.
        """
        self.to(self.device)

        df, exercise_map, num_exercises = self.preprocess_data(df)
        self.exercise_map = exercise_map
        self.num_exercises = num_exercises

        self.exercise_embedding = nn.Embedding(self.num_exercises + 1, self.d_model, padding_idx=0).to(self.device)

        train_dataset = SAKTDataset(df, self.exercise_map, self.num_steps, feature_col=self.feature_col)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=self.init_learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.learning_decay_rate)
        epochs_without_improvement = 0
        best_loss = 10

        for epoch in range(num_epochs):
            self.train()
            train_losses, all_labels, all_preds = [], [], []

            for past_exercises, past_responses, current_exercises, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                try:
                    past_exercises = past_exercises.to(self.device)
                    past_responses = past_responses.to(self.device)
                    current_exercises = current_exercises.to(self.device)
                    targets = targets.to(self.device)

                    optimizer.zero_grad()
                    preds = self(past_exercises, past_responses, current_exercises)
                    loss = self.compute_loss(preds, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    optimizer.step()

                    train_losses.append(loss.item())
                    all_labels.extend(targets.detach().cpu().numpy())
                    all_preds.extend(preds.detach().cpu().numpy())
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        optimizer.zero_grad()
                        gc.collect()
                        torch.cuda.empty_cache()
                        continue

            train_loss = np.mean(train_losses)
            train_auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.0
            print(f"Epoch {epoch+1}/{num_epochs}, Training loss: {train_loss:.4f}, AUC: {train_auc:.4f}")

            # Validation

            if train_loss > best_loss:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print("Early stopping triggered.")
                    break
            else:
                epochs_without_improvement = 0
                best_loss = train_loss

            prev_loss = train_loss

            scheduler.step()

    def evaluate(self, df_eval: pd.DataFrame, batch_size=64):
        """
        pretty bog standard eval, uses BCEWithLogits and creates a DS and DL using the provided eval df.
        """
        self.to(self.device)
        self.eval()

        df_eval = df_eval.dropna()
        df_eval[self.feature_col] = df_eval[self.feature_col].apply(lambda x: self.exercise_map.get(x, 0))

        eval_dataset = SAKTDataset(df_eval, self.exercise_map, self.num_steps, feature_col=self.feature_col)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

        val_losses, all_labels, all_preds = [], [], []

        with torch.no_grad():
            for past_exercises, past_responses, current_exercises, targets in tqdm(eval_loader, desc="Evaluating"):
                past_exercises = past_exercises.to(self.device)
                past_responses = past_responses.to(self.device)
                current_exercises = current_exercises.to(self.device)
                targets = targets.to(self.device)

                preds = self(past_exercises, past_responses, current_exercises)
                loss = self.compute_loss(preds, targets)

                val_losses.append(loss.item())
                all_labels.extend(targets.detach().cpu().numpy())
                all_preds.extend(preds.detach().cpu().numpy())

        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        all_pred_classes = np.round(all_preds)

        val_loss = np.mean(val_losses)
        val_auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.0
        val_f1 = f1_score(all_labels, all_pred_classes)
        print(f"Evaluation loss: {val_loss:.4f}, AUC: {val_auc:.4f}")
        return val_auc, val_loss

    def evaluate_internal(self, val_loader):
        """
       basically made this to not run the other method during train. does the same stuff without processing the df to
       make a dataloader and dataset.
        """
        self.eval()
        val_losses, all_labels, all_preds = [], [], []

        with torch.no_grad():
            for past_exercises, past_responses, current_exercises, targets in val_loader:
                past_exercises = past_exercises.to(self.device)
                past_responses = past_responses.to(self.device)
                current_exercises = current_exercises.to(self.device)
                targets = targets.to(self.device)

                preds = self(past_exercises, past_responses, current_exercises)
                loss = self.compute_loss(preds, targets)

                val_losses.append(loss.item())
                all_labels.extend(targets.detach().cpu().numpy())
                all_preds.extend(preds.detach().cpu().numpy())

        val_loss = np.mean(val_losses)
        val_auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.0
        return val_auc, val_loss
