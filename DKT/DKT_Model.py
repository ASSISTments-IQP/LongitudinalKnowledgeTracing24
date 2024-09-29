import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from sklearn.preprocessing import LabelEncoder
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import log_loss, roc_auc_score
from tqdm import tqdm


class DKT:
   def __init__(self):
      self.vocab_size = 0
      self.model = Sequential()

   def preprocess(self, data):
      self.vocab_size = data['old_problem_id'].nunique() + 1 # Number of unique problem IDs

      data = data[['user_xid', 'old_problem_id', 'skill_id', 'discrete_score', 'start_time']]
      data = data.sort_values(by=['user_xid', 'start_time'])
      data = data.fillna(0)  # Fill missing values with 0 for now

      le = LabelEncoder()
      data['encoded_problem_id'] = le.fit_transform(data['old_problem_id']) + 1 # Shift by 1 to reserve 0 for padding

      
      grouped = data.groupby('user_xid')

      seq = []
      lab = []
      for user, group in tqdm(grouped):
         group = group.sort_values(by='start_time')
         feature_seq = group['encoded_problem_id'].to_numpy()
         seq.append(torch.tensor(feature_seq, dtype=torch.float32))
         
         # Ensure labels are treated as a tensor
         labels = torch.tensor(group['discrete_score'].to_numpy(), dtype=torch.float32)
         lab.append(labels)

      # Padding sequences with zeros using PyTorch's pad_sequence
      padded_seq = pad_sequence(seq, batch_first=True, padding_value=0)  # (batch_size, timesteps)
      padded_lab = pad_sequence(lab, batch_first=True, padding_value=0.0)  # (batch_size, timesteps)

      # Reshape labels to have 3D shape (batch_size, timesteps, 1)
      padded_lab = padded_lab.unsqueeze(-1)

      return padded_seq, padded_lab
   

   def fit(self, data):
      print("Beginning data preprocessing")
      X, y = self.preprocess(data)
      print("Data preprocessing finished, beginning fitting.")

      self.model.add(Embedding(input_dim=self.vocab_size, output_dim=512, mask_zero=True, input_length=None))
      self.model.add(LSTM(64, activation='tanh', return_sequences=True))
      self.model.add(Dense(1, activation='sigmoid'))

      self.model.compile(optimizer='adam', 
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'AUC'])
      
      self.model.fit(X, y, 
         validation_data=(X, y),
         epochs=10, 
         batch_size=32)
      

      print("Model Training finished, final statistics:")
 
      
      print(f"Training loss: {self.model.history.history['loss'][-1]}")
      print(f"Training AUC: {self.model.history.history['AUC'][-1]}")

   

   def evaluate(self, data):
      if self.vocab_size == 0:
         print("Model has not been trained yet.")
         return
      
      X, y = self.preprocess(data)
      loss, accuracy, auc = self.model.evaluate(X, y)

      print("Evaulation Results")
      print(f"Test Loss: {loss}")
      print(f"Test Accuracy: {accuracy}")
      print(f"Test AUC: {auc}")



