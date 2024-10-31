import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Embedding, TimeDistributed, Dropout, StringLookup, IntegerLookup
import torch.nn.functional as F
import torch
from tqdm import tqdm
import keras
import math

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

			indices = np.array(y_true[:,0:25])
			y_true_rel = y_true[:,25:50]
			y_pred_rel = tf.Tensor(np.array(y_pred)[np.arange(y_true.shape[0]).astype(int),np.arange(y_true.shape[1]).astype(int),indices.astype(int)])
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
	def __init__(self, verbose=True, problem=False):
		self.model = None
		self.vocab = []
		self.vocab_size = 0
		self.num_dim = 0
		self.vocab_encoder = None
		self.label_encoder = None
		self.max_seq_len = 50
		self.verbose = verbose,
		if problem:
			self.vocab_col = 'old_problem_id'
		else:
			self.vocab_col = 'skill_id'


	def preprocess(self, data, fitting=False):
		data = data.sort_values(by=['user_xid', 'start_time'])
		data = data.fillna(0)  # Fill missing values with 0 for now

		if fitting:
			un = data[self.vocab_col].astype(str).unique()
			zer = un + '+0'
			on = un + '+1'

			self.vocab = np.concatenate([zer, on])
			self.vocab_size = len(self.vocab) + 2
			self.num_dim = math.ceil(math.log(self.vocab_size))
			self.vocab_encoder = StringLookup(vocabulary=self.vocab, output_mode='int', mask_token='na')
			self.label_encoder = IntegerLookup(vocabulary=data[self.vocab_col].unique(), mask_token=0, output_mode='int')

		data['vocab_id_x_correct'] = data[self.vocab_col].astype(str).copy()
		data.loc[data['discrete_score'] == 0, 'vocab_id_x_correct'] += '+0'
		data.loc[data['discrete_score'] == 1, 'vocab_id_x_correct'] += '+1'

		data['encoded_vocab_id'] = self.vocab_encoder(data['vocab_id_x_correct'])
		data['encoded_label_id'] = self.label_encoder(data[self.vocab_col])

		grouped = data.groupby('user_xid')

		f_seq = []
		l_seq = []
		lab = []
		for user, group in tqdm(grouped, disable=not self.verbose):
			group = group.sort_values(by='start_time')
			feature_seq = group['encoded_vocab_id'].to_numpy()
			correct_seq = group['discrete_score'].to_numpy()
			label_seq = group['encoded_label_id'].to_numpy()

			for start_idx in range(0, len(feature_seq), self.max_seq_len):
				end_idx = min(start_idx + self.max_seq_len, len(feature_seq))

				# Get subsequence for this user
				sub_feature_seq = feature_seq[start_idx:end_idx]
				sub_correct_seq = correct_seq[start_idx:end_idx]
				sub_label_seq = label_seq[start_idx:end_idx]

				# Pad feature sequence to max_seq_len
				padded_feature_seq = F.pad(torch.tensor(sub_feature_seq, dtype=torch.int64),
										   (0, self.max_seq_len - len(sub_feature_seq)),
										   value=0)
				f_seq.append(padded_feature_seq)

				padded_label_seq = F.pad(torch.tensor(sub_label_seq, dtype=torch.int64),
										 (0, self.max_seq_len - len(sub_label_seq)),
										 value=0)
				l_seq.append(padded_label_seq)

				padded_correct_seq = F.pad(torch.tensor(sub_correct_seq,dtype=torch.int8),
										 (0,self.max_seq_len - len(sub_correct_seq)),
										   value=-1)
				lab.append(padded_correct_seq)


				# Pad label sequence with shape [timesteps, vocab_size]
				# blank_labels = np.full((self.max_seq_len, self.vocab_size), -1, dtype=np.int8)
				# blank_labels[:len(sub_feature_seq), sub_feature_seq] = sub_correct_seq

		# Convert f_seq and lab to tensors
		f_seq = torch.stack(f_seq)
		l_seq = torch.stack(l_seq)
		lab = torch.stack(lab)
		lab = torch.hstack([l_seq,lab])

		return f_seq, lab


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
