# ORIGINAL AUTHOR 2021/4/23 @ zengxiaonan
import logging
import gc
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import os
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, log_loss, f1_score


def setup_multiprocessing():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass 


class DKTDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = [seq.astype(np.float32) if isinstance(seq, np.ndarray) else seq for seq in sequences]
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.sequences[idx]).float()


class Net(nn.Module):
    def __init__(self, num_questions, hidden_size, num_layers, dropout_rate):
        super(Net, self).__init__()
        self.hidden_dim = hidden_size
        self.layer_dim = num_layers
        self.lstm = nn.LSTM(
            num_questions * 2, hidden_size, num_layers, batch_first=True
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.hidden_dim, num_questions)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out)
        res = self.fc(out)  
        return res


def process_raw_pred(raw_question_matrix, raw_pred, num_questions: int) -> tuple:
    questions = torch.nonzero(raw_question_matrix)[1:, 1] % num_questions
    length = questions.shape[0]
    pred = raw_pred[:length]
    pred = pred.gather(1, questions.view(-1, 1)).flatten()
    truth = torch.nonzero(raw_question_matrix)[1:, 1] // num_questions
    del questions
    return pred, truth


class DKT:
    def __init__(
        self,
        batch_size=64,
        num_steps=50,
        hidden_size=128,
        lr=1e-4,
        dropout_rate=0.15,
        reg_lambda=1e-3,
        gpu_num=0,
        patience=5,
        num_workers=0,
        gradient_accumulation_steps=1,
        use_mixed_precision=True,
    ):
        self.vocab = []
        self.vocab_size = 0
        self.enc_dict = {}
        self.num_steps = num_steps
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.lr = lr
        self.batch_size = batch_size
        self.reg_lambda = reg_lambda
        self.dropout_rate = dropout_rate
        self.dkt_model = None
        self.patience = patience
        self.num_workers = num_workers
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_mixed_precision = use_mixed_precision
        
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass 
            
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.scaler = GradScaler() if self.use_mixed_precision and torch.cuda.is_available() else None

    def preprocess(self, df, fitting=False):
        if fitting:
            self.vocab = df["skill_id"].unique().tolist()
            self.vocab_size = len(self.vocab) + 1
            self.enc_dict = {sk_id: i for i, sk_id in enumerate(self.vocab, start=1)}

        df.drop_duplicates("problem_log_id", inplace=True)
        df.sort_values(by=["user_xid", "start_time"], inplace=True)


        def sequence_generator():
            for name, group in df.groupby(by="user_xid"):
                group_len = group.shape[0]

                if group_len < 5:
                    continue
                    
                mod = (
                    0
                    if group_len % self.num_steps == 0
                    else (self.num_steps - group_len % self.num_steps)
                )

                oh = np.zeros(shape=(group_len + mod, self.vocab_size * 2), dtype=np.float32)
                
                for i, (idx, row) in enumerate(group.iterrows()):
                    if i >= group_len:
                        break
                    skill = row["skill_id"]
                    corr = row["discrete_score"]
                    found_vocab = self.check_vocab(skill)
                    col_idx = found_vocab if corr == 0 else found_vocab + self.vocab_size
                    oh[i][col_idx] = 1

                seq_reshaped = oh.reshape(-1, self.num_steps, 2 * self.vocab_size)
                for seq_chunk in seq_reshaped:
                    yield seq_chunk
   
                del oh, seq_reshaped

        sequences_list = []
        batch_count = 0
        for seq in tqdm(sequence_generator(), desc="Processing sequences"):
            sequences_list.append(seq)
            batch_count += 1

            if batch_count >= 1000:  
                break

        gc.collect()
        
        dataset = DKTDataset(sequences_list)
        d_l = DataLoader(
            dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,  
            persistent_workers=False,  
            drop_last=True
        )
        return d_l

    def fit(self, train_data, num_epochs):
        train_data = self.preprocess(train_data, fitting=True)
        self.dkt_model = Net(
            self.vocab_size, self.hidden_size, self.num_layers, self.dropout_rate
        )
        self.dkt_model.to(self.device)
        loss_function = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.dkt_model.parameters(), lr=self.lr)

        best_loss = float('inf')
        pat_count = 0
        for e in range(num_epochs):
            self.dkt_model.train()
            epoch_losses = []
            all_pred, all_target = [], []
            optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(tqdm(train_data, desc=f"Epoch {e}")):
                try:
                    batch = batch.to(self.device, non_blocking=True)
                    if self.scaler is not None:
                        with autocast():
                            integrated_pred = self.dkt_model(batch)
                            batch_losses = []
                            
                            batch_size = batch.shape[0]
                            for student in range(batch_size):
                                pred, truth = process_raw_pred(
                                    batch[student], integrated_pred[student], self.vocab_size
                                )
                                if len(pred) > 0: 
                                    loss = loss_function(pred, truth.float())
                                    batch_losses.append(loss)
                                    pred_probs = torch.sigmoid(pred)
                                    all_pred.append(pred_probs.detach().cpu())
                                    all_target.append(truth.detach().cpu().float())
                                
                                del pred, truth
                            
                            if batch_losses:
                                avg_batch_loss = torch.stack(batch_losses).mean()
                                avg_batch_loss = avg_batch_loss / self.gradient_accumulation_steps
                                epoch_losses.append(avg_batch_loss.item() * self.gradient_accumulation_steps)
                        
                        if batch_losses:
                            self.scaler.scale(avg_batch_loss).backward()
                    else:
                        integrated_pred = self.dkt_model(batch)
                        batch_losses = []
                        
                        batch_size = batch.shape[0]
                        for student in range(batch_size):
                            pred, truth = process_raw_pred(
                                batch[student], integrated_pred[student], self.vocab_size
                            )
                            if len(pred) > 0:
                                loss = loss_function(pred, truth.float())
                                batch_losses.append(loss)
                                pred_probs = torch.sigmoid(pred)
                                all_pred.append(pred_probs.detach().cpu())
                                all_target.append(truth.detach().cpu().float())
                            
                            del pred, truth
                        
                        if batch_losses:
                            avg_batch_loss = torch.stack(batch_losses).mean()
                            avg_batch_loss = avg_batch_loss / self.gradient_accumulation_steps
                            epoch_losses.append(avg_batch_loss.item() * self.gradient_accumulation_steps)
                            avg_batch_loss.backward()
                    
                    del batch, integrated_pred
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        if self.scaler is not None:
                            self.scaler.step(optimizer)
                            self.scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad()
                    if batch_idx % 50 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                except RuntimeError as error:
                    if "out of memory" in str(error):
                        print(f"OOM error at batch {batch_idx}, clearing cache and continuing...")
                        optimizer.zero_grad()
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise error
            if len(train_data) % self.gradient_accumulation_steps != 0:
                if self.scaler is not None:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            if epoch_losses:
                avg_epoch_loss = np.mean(epoch_losses)
                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
                    pat_count = 0
                else:
                    pat_count += 1
                    if pat_count >= self.patience:
                        print(f"Early stopping after {self.patience} epochs without improvement")
                        break
                
                print(f"[Epoch {e}] Average Loss: {avg_epoch_loss:.6f}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if all_pred and all_target:
            all_pred = torch.cat(all_pred)
            all_target = torch.cat(all_target)
            return roc_auc_score(all_target.detach().numpy(), all_pred.detach().numpy())
        else:
            return 0.0

    def evaluate(self, test_data) -> tuple:
        if self.dkt_model is not None:
            test_data = self.preprocess(test_data)
            self.dkt_model.eval()
            y_pred = []
            y_truth = []
            
            with torch.no_grad(): 
                for batch_idx, batch in enumerate(tqdm(test_data, desc="Evaluating")):
                    try:
                        batch = batch.to(self.device, non_blocking=True)
    
                        if self.scaler is not None:
                            with autocast():
                                integrated_pred = self.dkt_model(batch)
                        else:
                            integrated_pred = self.dkt_model(batch)
                        
                        batch_size = batch.shape[0]
                        for student in range(batch_size):
                            pred, truth = process_raw_pred(
                                batch[student], integrated_pred[student], self.vocab_size
                            )
                            if len(pred) > 0:
                                pred_probs = torch.sigmoid(pred)
                                y_pred.append(pred_probs.cpu())
                                y_truth.append(truth.cpu().float())
                            del pred, truth
                        
                        del batch, integrated_pred

                        if batch_idx % 50 == 0:
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                
                    except RuntimeError as error:
                        if "out of memory" in str(error):
                            print(f"OOM error during evaluation at batch {batch_idx}, clearing cache...")
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue
                        else:
                            raise error

            if not y_pred or not y_truth:
                return 0.0, float('inf'), 0.0
                
            y_pred = torch.cat(y_pred)
            y_truth = torch.cat(y_truth)
            y_truth = y_truth.detach().numpy()
            y_pred = y_pred.detach().numpy()
            y_pred_class = np.round(y_pred)

            auc = roc_auc_score(y_truth, y_pred)
            ll = log_loss(y_truth, y_pred)
            f1 = f1_score(y_truth, y_pred_class)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return auc, ll, f1
        else:
            return 0.0, float('inf'), 0.0
        

    def check_vocab(self, key):
        return self.enc_dict.get(key, 0)

    def save(self, filepath):
        if self.dkt_model is not None:
            torch.save(self.dkt_model.state_dict(), filepath)
            logging.info("save parameters to %s" % filepath)
        else:
            logging.warning("No model to save")

    def load(self, filepath):
        if self.dkt_model is not None:
            state_dict = torch.load(filepath, map_location='cuda' if torch.cuda.is_available() else 'cpu')
            self.dkt_model.load_state_dict(state_dict)
            self.dkt_model.to(self.device)
            logging.info("load parameters from %s" % filepath)
        else:
            logging.warning("No model initialized to load parameters into")