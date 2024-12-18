import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Embedding, TimeDistributed, Dropout, StringLookup
import torch.nn.functional as F
import torch
from tqdm import tqdm
import keras
import math

tf.config.run_functions_eagerly(True)


class DKT_model(tf.keras.Model):
	def __init__(self, vocab_size, num_dim, max_seq_len=50, verbose=True):
		input = tf.keras.Input(shape=(max_seq_len,))
		emb = Embedding(input_dim=vocab_size, output_dim=num_dim, mask_zero=True)(input)
		x = LSTM(64, activation='tanh', return_sequences=True)(emb)
		dr = Dropout(0.33)(x)
		output = TimeDistributed(Dense(vocab_size, activation='sigmoid'))(dr)
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

		super(DKT_model, self).compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
									   loss=custom_loss,
									   metrics=['accuracy', 'AUC'])

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


	def fit(self, data, num_epochs=3):
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


	def evaluate(self, data):
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