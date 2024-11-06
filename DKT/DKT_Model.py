import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
import math


# no verbose for now
class DKT_model(nn.Module):
	def __init__(self, vocab_size, embedding_dim, hidden_dim):
		super(DKT_model, self).__init__()
		self.emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
		self.dr = nn.Dropout(0.2)
		self.ltsm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
		self.fc = nn.Linear(hidden_dim, vocab_size)
		self.sigmoid = nn.Sigmoid
  
  
	def forward(self, x):
		x = self.emb(x)
		x = self.dr(x)
		output, _ = self.ltsm(x)
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
            u_data['skill_id_n_corr'] = u_data['skill_id']
            u_data.loc[u_data['discrete_score'] == 0, 'skill_id_n_corr'] += '0'
            u_data.loc[u_data['discrete_score'] == 1, 'skill_id_n_corr'] += '1'
            encoded_seq = [self.vocab_to_idx.get(s, self.vocab_to_idx['<UNK>']) for s in u_data['skill_id_x_corr']]
            correct_seq = u_data['discrete_score'].to_numpy()
            
            seq_len = len(encoded_seq)
            for start_idx in range(0, seq_len, self.max_seq_len):
                end_idx = min(start_idx + self.max_seq_len, seq_len)
                sub_feat_seq = encoded_seq[start_idx:end_idx]
                sub_correct_seq = correct_seq[start_idx:end_idx]
                #pad thaim
                pad_len = self.max_seq_len -len(sub_feat_seq)
                input_seq = sub_feat_seq + [self.vocab_to_idx['<PAD>']] * pad_len
                
                label_seq = np.full((self.max_seq_len, len(self.vocab_to_idx)), -1 , dtype = np.float32)
                for i, (enc, corr) in enumerate(zip(sub_feat_seq, sub_correct_seq)):
                    label_seq[i, enc] = corr
                
                self.samples.append((input_seq, label_seq))
    def __len__(self):
        return len(self.samples)
    def __get_item__(self, idx):
        input_seq, label = self.samples[idx]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(label, dtype = torch.float32)
    
    

	@tf.function
	def fit(self, x, y, epochs, batch_size):
		return super(DKT_model, self).fit(x=x, y=y, epochs=epochs, batch_size=batch_size, verbose = self.verbose)

	def eval(self, x, y):
		return super(DKT_model, self).evaluate(x=x, y=y, verbose = self.verbose)


class DKT:
	def __init__(self, verbose=True):
		self.model = None
		self.vocab = []
		self.vocab_size = 0
		self.num_dim = 0
		self.vocab_encoder = None
		self.max_seq_len = 50
		self.verbose = verbose


	def preprocess(self, data, fitting=False):
		data = data.sort_values(by=['user_xid', 'start_time'])
		data = data.fillna(0)  # Fill missing values with 0 for now

		if fitting:
			un = data['skill_id'].astype(str).unique()
			zer = un + '+0'
			on = un + '+1'

			self.vocab = np.concatenate([zer, on])
			self.vocab_size = len(self.vocab) + 2
			self.num_dim = math.ceil(math.log(self.vocab_size))
			self.vocab_encoder = StringLookup(vocabulary=self.vocab, output_mode='int', mask_token='na')

		data['skill_id_x_correct'] = data['skill_id'].astype(str).copy()
		data.loc[data['discrete_score'] == 0, 'skill_id_x_correct'] += '+0'
		data.loc[data['discrete_score'] == 1, 'skill_id_x_correct'] += '+1'

		data['encoded_problem_id'] = self.vocab_encoder(data['skill_id_x_correct'])

		grouped = data.groupby('user_xid')

		seq = []
		lab = []
		for user, group in tqdm(grouped, disable=not self.verbose):
			group = group.sort_values(by='start_time')
			feature_seq = group['encoded_problem_id'].to_numpy()
			correct_seq = group['discrete_score'].to_numpy()

			for start_idx in range(0, len(feature_seq), self.max_seq_len):
				end_idx = min(start_idx + self.max_seq_len, len(feature_seq))

				# Get subsequence for this user
				sub_feature_seq = feature_seq[start_idx:end_idx]
				sub_correct_seq = correct_seq[start_idx:end_idx]

				# Pad feature sequence to max_seq_len
				padded_feature_seq = F.pad(torch.tensor(sub_feature_seq, dtype=torch.int8),
										   (0, self.max_seq_len - len(sub_feature_seq)),
										   value=0)
				seq.append(padded_feature_seq)

				# Pad label sequence with shape [timesteps, vocab_size]
				blank_labels = np.full((self.max_seq_len, self.vocab_size), -1, dtype=np.int8)
				blank_labels[:len(sub_feature_seq), sub_feature_seq] = sub_correct_seq

				lab.append(torch.tensor(blank_labels, dtype=torch.int8))

		# Convert seq and lab to tensors
		seq = torch.stack(seq)
		lab = torch.stack(lab)

		return seq, lab


	def fit(self, data, num_epochs=5):
		if self.verbose:
			print("Beginning data preprocessing")
		X, y = self.preprocess(data, fitting=True)

		self.model = DKT_model(self.vocab_size, self.num_dim, self.max_seq_len, self.verbose)
		if self.verbose:
			print("Data preprocessing finished, beginning fitting.")
		self.model.compile()

		self.model.fit(X, y,
					   epochs=num_epochs,
					   batch_size=64)

		print("Model Training finished, final statistics:")
		loss, acc, auc = self.model.eval(X, y)
		print(f"Training loss: {loss}")
		print(f"Training AUC: {auc}")

		return auc


	def eval(self, data):
		if self.vocab_size == 0:
			print("Model has not been trained yet.")
			return

		X, y = self.preprocess(data)
		loss, accuracy, auc = self.model.eval(X, y)

		print("Evaulation Results")
		print(f"Test Loss: {loss}")
		print(f"Test Accuracy: {accuracy}")
		print(f"Test AUC: {auc}")

		return auc
