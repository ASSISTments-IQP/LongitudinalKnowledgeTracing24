import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Generator, Dict
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score


class SAKTModel(nn.Module):
    def __init__(self, num_steps: int = 50, batch_size: int = 16, d_model: int = 128,
                 num_heads: int = 8, dropout_rate: float = 0.2, device='cuda'):
        super(SAKTModel, self).__init__()
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.exercise_map = {}
        self.num_exercises = 0

        # Define model layers
        self.exercise_embedding = nn.Embedding(self.num_exercises + 1, self.d_model)
        self.response_embedding = nn.Embedding(2, self.d_model)
        self.position_embedding = nn.Embedding(self.num_steps, self.d_model)
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=self.num_heads, dropout=self.dropout_rate)
        
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.layer_norm1 = nn.LayerNorm(self.d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 4),
            nn.ReLU(),
            nn.Linear(self.d_model * 4, self.d_model)
        )
        
        self.dropout2 = nn.Dropout(self.dropout_rate)
        self.layer_norm2 = nn.LayerNorm(self.d_model)
        self.output_layer = nn.Linear(self.d_model, 1)
        
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.to(self.device)
        print(f"Using device: {self.device}")

    def _data_generator(self, df: pd.DataFrame) -> Generator[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
        df = df.copy()
        df['start_time'] = pd.to_datetime(df['start_time'], utc=True)
        df = df[['user_xid', 'old_problem_id', 'discrete_score', 'start_time']].sort_values(
            by=['user_xid', 'start_time'])
        user_xids = df['user_xid'].unique()

        batch_past_exercises, batch_past_responses, batch_current_exercises, batch_targets = [], [], [], []

        for user_xid in user_xids:
            user_data = df[df['user_xid'] == user_xid].sort_values('start_time')
            exercise_seq = [self.exercise_map.get(id, 0) for id in user_data['old_problem_id']]
            response_seq = user_data['discrete_score'].astype(int).tolist()

            for idx in range(1, len(exercise_seq)):
                start_idx = max(0, idx - self.num_steps)
                past_exercises = exercise_seq[start_idx:idx]
                past_responses = response_seq[start_idx:idx]
                current_exercise = exercise_seq[idx]
                target_response = response_seq[idx]

                pad_len = self.num_steps - len(past_exercises)
                past_exercises = [0] * pad_len + past_exercises
                past_responses = [0] * pad_len + past_responses

                batch_past_exercises.append(past_exercises)
                batch_past_responses.append(past_responses)
                batch_current_exercises.append(current_exercise)
                batch_targets.append(target_response)

                if len(batch_past_exercises) == self.batch_size:
                    yield (
                        torch.tensor(batch_past_exercises, dtype=torch.long, device=self.device),
                        torch.tensor(batch_past_responses, dtype=torch.long, device=self.device),
                        torch.tensor(batch_current_exercises, dtype=torch.long, device=self.device),
                        torch.tensor(batch_targets, dtype=torch.float32, device=self.device)
                    )
                    batch_past_exercises, batch_past_responses, batch_current_exercises, batch_targets = [], [], [], []

        if batch_past_exercises:
            pad_size = self.batch_size - len(batch_past_exercises)
            if pad_size > 0:
                pad_exercises = [[0] * self.num_steps] * pad_size
                pad_responses = [[0] * self.num_steps] * pad_size
                pad_current_exercises = [0] * pad_size
                pad_targets = [0.0] * pad_size

                batch_past_exercises.extend(pad_exercises)
                batch_past_responses.extend(pad_responses)
                batch_current_exercises.extend(pad_current_exercises)
                batch_targets.extend(pad_targets)

            yield (
                torch.tensor(batch_past_exercises, dtype=torch.long, device=self.device),
                torch.tensor(batch_past_responses, dtype=torch.long, device=self.device),
                torch.tensor(batch_current_exercises, dtype=torch.long, device=self.device),
                torch.tensor(batch_targets, dtype=torch.float32, device=self.device)
            )

    def preprocess(self, df: pd.DataFrame) -> None:
        unique_problems = df['old_problem_id'].unique()
        self.exercise_map = {problem: idx for idx, problem in enumerate(unique_problems, start=1)}
        self.num_exercises = len(self.exercise_map)
        self.exercise_embedding = nn.Embedding(self.num_exercises + 1, self.d_model)
        print("Model layers initialized.")

    def forward(self, past_exercises, past_responses, current_exercises):
        past_exercise_embeddings = self.exercise_embedding(past_exercises)  # [batch_size, seq_len, d_model]
        past_response_embeddings = self.response_embedding(past_responses)  # [batch_size, seq_len, d_model]
        past_interactions_embeddings = past_exercise_embeddings + past_response_embeddings
        seq_len = past_exercises.size(1)
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)
        position_embeddings = self.position_embedding(position_ids)  # [1, seq_len, d_model]
        past_interactions_embeddings += position_embeddings
        attention_mask = (past_exercises != 0).unsqueeze(1).repeat(1, self.num_heads, 1)  # [batch_size, num_heads, seq_len]
        attention_mask = attention_mask.view(-1, 1, seq_len)  # Reshape to [batch_size * num_heads, 1, seq_len]
        query = self.exercise_embedding(current_exercises).unsqueeze(1)  # [batch_size, 1, d_model]
        key = value = past_interactions_embeddings.permute(1, 0, 2)  # [seq_len, batch_size, d_model]
        query = query.permute(1, 0, 2)  # [1, batch_size, d_model]
        attention_output, _ = self.multi_head_attention(query, key, value, attn_mask=attention_mask)
        attention_output = attention_output.permute(1, 0, 2)  # [batch_size, 1, d_model]
        out1 = self.layer_norm1(attention_output + query.permute(1, 0, 2))
        ffn_output = self.ffn(out1)
        out2 = self.layer_norm2(ffn_output + out1)
        output = self.output_layer(out2).squeeze(-1)
        return output

    def fit(self, train_df, val_df=None, num_epochs=5, early_stopping=True, patience=1):
        self.preprocess(train_df)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.96)

        best_val_auc = 0.0
        epochs_since_improvement = 0

        for epoch in range(num_epochs):
            self.train()
            train_losses, all_labels, all_predictions = [], [], []

            train_data = self._data_generator(train_df)
            for batch in tqdm(train_data, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100):
                past_exercises, past_responses, current_exercises, y = batch
                y = y.float()
                self.optimizer.zero_grad()
                
                predictions = self(past_exercises, past_responses, current_exercises)
                # if torch.isnan(predictions).any():
                #     print("NaN detected in predictions!")
                #     continue
                # print(f"Predictions: {predictions}")  # Check values are in [0, 1]
                loss = self.loss_fn(predictions.squeeze(-1), y)
                loss.backward()
                self.optimizer.step()

                train_losses.append(loss.item())
                all_labels.extend(y.cpu().numpy())
                all_predictions.extend(predictions.detach().cpu().numpy())

            epoch_auc = roc_auc_score(all_labels, all_predictions)
            print(f"Epoch {epoch + 1} - Loss: {np.mean(train_losses):.4f}, AUC: {epoch_auc:.4f}")

            # Validation and early stopping
            if val_df is not None:
                val_auc, val_loss = self.eval(val_df)
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    epochs_since_improvement = 0
                    torch.save(self.state_dict(), 'best_model_weights.pth')
                else:
                    epochs_since_improvement += 1
                if early_stopping and epochs_since_improvement >= patience:
                    print("Early stopping triggered.")
                    break

            self.scheduler.step()

    def eval(self, val_df):
        self.eval()
        val_losses, all_labels, all_predictions = [], [], []
        val_data = self._data_generator(val_df)

        with torch.no_grad():
            for batch in val_data:
                past_exercises, past_responses, current_exercises, y = batch
                predictions = self(past_exercises, past_responses, current_exercises)
                loss = self.loss_fn(predictions, y)
                val_losses.append(loss.item())
                all_labels.extend(y.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())

        val_auc = roc_auc_score(all_labels, all_predictions)
        val_loss = np.mean(val_losses)
        return val_auc, val_loss
