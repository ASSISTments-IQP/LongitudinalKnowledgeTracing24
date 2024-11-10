import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Embedding, TimeDistributed, Dropout, StringLookup
import torch.nn.functional as F
import torch
from tqdm import tqdm
import keras
import math
from torch import nn
from torch.optim import Adam

tf.config.run_functions_eagerly(True)


class DKT_model(tf.keras.Model):
	def __init__(self, vocab_size, num_dim, max_seq_len, verbose=True):
		input = tf.keras.Input(shape=(max_seq_len,))
		emb = Embedding(input_dim=vocab_size, output_dim=num_dim, mask_zero=True)(input)
		dr = Dropout(0.2)(emb)
		x = LSTM(124, activation='tanh', return_sequences=True)(dr)
		output = TimeDistributed(Dense(vocab_size, activation='sigmoid'))(x)
		if verbose:
			self.verbose = 2
		else:
			self.verbose = 0
		super(DKT_model, self).__init__(inputs=input, outputs=output)

	def compile(self):
		def custom_loss(y_true, y_pred):
			indices = tf.math.not_equal(y_true, -1)
			y_true_rel = y_true[indices]
			y_pred_rel = y_pred[indices]
			return tf.keras.losses.binary_crossentropy(y_true_rel, y_pred_rel)

		super(DKT_model, self).compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
									   loss=custom_loss,
									   metrics=['accuracy', 'AUC'])

	@tf.function
	def fit(self, x, y, epochs, batch_size):
		return super(DKT_model, self).fit(x=x, y=y, epochs=epochs, batch_size=batch_size, verbose = self.verbose)

	def eval(self, x, y):
		return super(DKT_model, self).evaluate(x=x, y=y, verbose = self.verbose)


class DKT:
	def __init__(self, max_seq_len, batch_size, verbose=True):
		self.model = None
		self.vocab = []
		self.vocab_size = 0 # Determined by data
		self.num_dim = 0 # Determined by data
		#self.vocab_encoder = None
		self.max_seq_len = max_seq_len
		self.batch_size = batch_size
		self.verbose = verbose



	def _data_generator(self, data):
				df = pd.DataFrame(data)
				df = df[['user_xid', 'skill_id', 'discrete_score', 'start_time']].sort_values(by=['user_xid', 'start_time'])
				user_xids = df['user_xid'].unique()

				batch_feature_seqs, batch_label_seqs = [], []

				for user_xid in tqdm(user_xids, disable=not self.verbose):
					user_data = df[df['user_xid'] == user_xid].sort_values('start_time')
					skill_seq = user_data['skill_id'].tolist()
					score_seq = user_data['discrete_score'].astype(int).tolist()

					for idx in range(1, len(skill_seq)):
							start_idx = max(0, idx - self.max_seq_len)
							past_skills = skill_seq[start_idx:idx]
							past_scores = score_seq[start_idx:idx]

							# Ensure sequences are within the desired length
							if len(past_skills) > self.max_seq_len:
								past_skills = past_skills[-self.max_seq_len:]
								past_scores = past_scores[-self.max_seq_len:]

							# Pad sequences to the fixed length `num_steps`
							pad_len = self.max_seq_len - len(past_skills)
							past_skills = [0] * pad_len + past_skills
							past_scores = [-1] * pad_len + past_scores  # Use -1 for padding labels

							batch_feature_seqs.append(past_skills)
							batch_label_seqs.append(past_scores)

							# Yield a batch when enough data has been collected
							if len(batch_feature_seqs) == self.batch_size:

								yield (
									torch.tensor(batch_feature_seqs, dtype=torch.int32),
									torch.tensor(batch_label_seqs, dtype=torch.int32)
								)
								batch_feature_seqs, batch_label_seqs = [], []

				# Handle any remaining sequences that didn't make a full batch
				if batch_feature_seqs:
					pad_size = self.batch_size - len(batch_feature_seqs)
					if pad_size > 0:
							# Pad remaining sequences to create a full batch
							pad_skills = [[0] * self.max_seq_len] * pad_size
							pad_scores = [[-1] * self.max_seq_len] * pad_size

							batch_feature_seqs.extend(pad_skills)
							batch_label_seqs.extend(pad_scores)

					yield (
							torch.tensor(batch_feature_seqs, dtype=torch.int32),
							torch.tensor(batch_label_seqs, dtype=torch.int32)
					)




	def fitGen(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None, num_epochs: int = 5, early_stopping: bool = True, patience: int = 1):
			optimizer = Adam()
			loss_fn = nn.BCEWithLogitsLoss()
			best_val_auc = 0.0
			epochs_since_improvement = 0

			for epoch in range(num_epochs):
				train_loss = 0.0
	
				train_data = self._data_generator(train_df)
				total_batches = (len(train_df) + self.batch_size - 1) // self.batch_size
				train_pbar = tqdm.tqdm(train_data, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", total=total_batches, ncols=100)

				for past_skills, past_scores in train_pbar:
						optimizer.zero_grad()
						predictions = self.model(past_skills)  # Replace with your model’s forward method

						mask = past_scores != -1
						loss = loss_fn(predictions[mask], past_scores[mask].float())
						loss.backward()
						optimizer.step()

						train_loss += loss.item()
						train_pbar.set_postfix({'loss': f'{train_loss / (total_batches):.4f}'})

				print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {train_loss / total_batches:.4f}")

				if val_df is not None:
						val_metrics = self.eval(val_df)  # Replace with your evaluation method
						val_auc_score = val_metrics['auc']  # Assumes eval method returns 'auc'

						if val_auc_score > best_val_auc:
							best_val_auc = val_auc_score
							epochs_since_improvement = 0
							torch.save(self.model.state_dict(), 'best_model_weights.pth')
						else:
							epochs_since_improvement += 1

						if early_stopping and epochs_since_improvement >= patience:
							print(f"Early stopping triggered after {epoch + 1} epochs.")
							break

			if val_df is not None and early_stopping:
				self.model.load_state_dict(torch.load('best_model_weights.pth'))


	# def preprocess(self, data, fitting=False):
	# 	data = data.sort_values(by=['user_xid', 'start_time'])
	# 	data = data.fillna(0)  # Fill missing values with 0 for now

	# 	if fitting:
	# 		un = data['skill_id'].astype(str).unique()
	# 		zer = un + '+0'
	# 		on = un + '+1'

	# 		self.vocab = np.concatenate([zer, on])
	# 		self.vocab_size = len(self.vocab) + 2
	# 		self.num_dim = math.ceil(math.log(self.vocab_size))
	# 		self.vocab_encoder = StringLookup(vocabulary=self.vocab, output_mode='int', mask_token='na')

	# 	data['skill_id_x_correct'] = data['skill_id'].astype(str).copy()
	# 	data.loc[data['discrete_score'] == 0, 'skill_id_x_correct'] += '+0'
	# 	data.loc[data['discrete_score'] == 1, 'skill_id_x_correct'] += '+1'

	# 	data['encoded_problem_id'] = self.vocab_encoder(data['skill_id_x_correct'])

	# 	grouped = data.groupby('user_xid')

	# 	seq = []
	# 	lab = []
	# 	for user, group in tqdm(grouped, disable=not self.verbose):
	# 		group = group.sort_values(by='start_time')
	# 		feature_seq = group['encoded_problem_id'].to_numpy()
	# 		correct_seq = group['discrete_score'].to_numpy()

	# 		for start_idx in range(0, len(feature_seq), self.max_seq_len):
	# 			end_idx = min(start_idx + self.max_seq_len, len(feature_seq))

	# 			# Get subsequence for this user
	# 			sub_feature_seq = feature_seq[start_idx:end_idx]
	# 			sub_correct_seq = correct_seq[start_idx:end_idx]

	# 			# Pad feature sequence to max_seq_len
	# 			padded_feature_seq = F.pad(torch.tensor(sub_feature_seq, dtype=torch.int8),
	# 									   (0, self.max_seq_len - len(sub_feature_seq)),
	# 									   value=0)
	# 			seq.append(padded_feature_seq)

	# 			# Pad label sequence with shape [timesteps, vocab_size]
	# 			blank_labels = np.full((self.max_seq_len, self.vocab_size), -1, dtype=np.int8)
	# 			blank_labels[:len(sub_feature_seq), sub_feature_seq] = sub_correct_seq

	# 			lab.append(torch.tensor(blank_labels, dtype=torch.int8))

	# 	# Convert seq and lab to tensors
	# 	seq = torch.stack(seq)
	# 	lab = torch.stack(lab)

	# 	return seq, lab


	def fit(self, data, num_epochs=5):
		if self.verbose:
			print("Beginning data preprocessing")
		X, y = self._data_generator(data)

		self.model = DKT_model(self.vocab_size, self.num_dim, self.max_seq_len, self.verbose)
		if self.verbose:
			print("Data preprocessing finished, beginning fitting.")
		self.model.compile()

		# self.model.fit(X, y,
		# 			   epochs=num_epochs,
		# 			   batch_size=64)

		self.model.fitGen(data, num_epochs=num_epochs)

		print("Model Training finished, final statistics:")
		loss, acc, auc = self.model.eval(X, y)
		print(f"Training loss: {loss}")
		print(f"Training Acc: {acc}")
		print(f"Training AUC: {auc}")

		return auc


	def eval(self, data):
		if self.vocab_size == 0:
			print("Model has not been trained yet.")
			return

		# X, y = self.preprocess(data)
		X, y = self._data_generator(data)
		loss, accuracy, auc = self.model.eval(X, y)

		print("Evaulation Results")
		print(f"Test Loss: {loss}")
		print(f"Test Accuracy: {accuracy}")
		print(f"Test AUC: {auc}")

		return auc
