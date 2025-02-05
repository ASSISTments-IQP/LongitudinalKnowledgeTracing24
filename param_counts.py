import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
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


class DKTNet(nn.Module):
	def __init__(self, num_questions, hidden_size, num_layers):
		super(DKTNet, self).__init__()
		self.hidden_dim = hidden_size
		self.layer_dim = num_layers
		self.lstm = nn.LSTM(num_questions * 2, hidden_size, num_layers, batch_first=True)
		self.fc = nn.Linear(self.hidden_dim, num_questions)

	def forward(self, x):
		out, _ = self.lstm(x)
		res = torch.sigmoid(self.fc(out))
		return res


class DSAKTModel(nn.Module):
	def __init__(self, num_steps=50, batch_size=32, d_model=128, num_heads=8, dropout_rate=0.2, init_learning_rate=1e-3,
				 learning_decay_rate=0.98, feature_col='skill_id', gpu_num=0):
		super(DSAKTModel, self).__init__()
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

		# os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)
		self.device = torch.device('cpu')

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
		batch_size = past_exercises.size(0)
		past_exercise_emb = self.exercise_embedding(past_exercises)  # Shape: (batch_size, seq_len, d_model)
		past_response_emb = self.response_embedding(past_responses)  # Shape: (batch_size, seq_len, d_model)
		interactions_emb = past_exercise_emb + past_response_emb  # Shape: (batch_size, seq_len, d_model)

		seq_len = past_exercises.size(1)
		position_ids = torch.arange(seq_len, device=past_exercises.device).unsqueeze(0)  # Shape: (1, seq_len)
		position_emb = self.position_embedding(position_ids)  # Shape: (1, seq_len, d_model)
		interactions_emb += position_emb  # Shape: (batch_size, seq_len, d_model)

		attention_mask = (past_exercises == float('-inf'))  # Shape: (batch_size, seq_len)

		query = self.exercise_embedding(current_exercises).unsqueeze(1).permute(1, 0,
																				2)  # Shape: (1, batch_size, d_model)
		key = value = interactions_emb.permute(1, 0, 2)  # Shape: (seq_len, batch_size, d_model)

		attn_output, _ = self.attention(query, key, value, key_padding_mask=attention_mask)
		attn_output = attn_output.permute(1, 0, 2)  # Shape: (batch_size, 1, d_model)
		out1 = self.norm1(attn_output + query.permute(1, 0, 2))  # Shape: (batch_size, 1, d_model)
		ffn_out = self.ffn(out1)  # Shape: (batch_size, 1, d_model)
		out2 = self.norm2(ffn_out + out1)  # Shape: (batch_size, 1, d_model)
		output = self.output_layer(out2).squeeze(-1).squeeze(-1)  # Shape: (batch_size)
		return output

	def compute_loss(self, predictions, targets):
		return self.loss_fn(predictions, targets)

	def fit(self, df: pd.DataFrame, batch_size=64, num_epochs=5, lr=1e-3, patience=2, validation_split=0.1):
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

		return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
	year_list = ['19-20', '20-21', '21-22', '22-23', '23-24']
	print('Loading year samples')
	sample_dict_all = {}
	for y in year_list:
		y_dict = {}
		for i in range(1, 11):
			s1 = pd.read_csv(f'./Data/samples/{y}/sample{i}.csv')
			y_dict[i] = s1
		sample_dict_all[y] = y_dict
	res = []

	for key0, val_d in sample_dict_all.items():
		for key, val in val_d.items():
			res.append(['BKT', (len(val.skill_id.unique()) + 1) * 5])
			res.append(['PFA', (len(val.skill_id.unique()) + 1) * 3 + 1])
			dummy_dkt = DKTNet((len(val.skill_id.unique()) + 1) * 2, 96, 1)
			res.append(['DKT', sum(p.numel() for p in dummy_dkt.parameters() if p.requires_grad)])
			dummy_sakt_e = DSAKTModel(60, 64, 352, 8, 0.43, 1e-4, 0.7, feature_col='old_problem_id')
			dummy_sakt_kc = DSAKTModel(100, 48, 128, 16, 0.188, 1e-4, 0.868, feature_col='skill_id')
			res.append(['SAKT-E', dummy_sakt_e.fit(val)])
			res.append(['SAKT-KC', dummy_sakt_kc.fit(val)])

	res_df = pd.DataFrame(res, columns=['model', 'num_trainable_params'])
	res_df.to_csv('./Data/param_numbers.csv')
