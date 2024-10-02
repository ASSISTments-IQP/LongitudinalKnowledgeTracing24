import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Masking
from sklearn.preprocessing import LabelEncoder
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import log_loss, roc_auc_score
from tqdm import tqdm
from keras.layers import Dropout
import math


class DKT:
   def __init__(self):
      self.vocab_size = 0
      self.model = Sequential()
      self.numDim = 0

   def preprocess(self, data):
      self.vocab_size = data['problem_id'].nunique() + 1 # Number of unique problem IDs
      self.numDim = math.ceil(math.log(self.vocab_size)) # Number of dimensions for embedding layer

      data = data[['user_id', 'problem_id', 'skill_id', 'correct', 'order_id']]
      data = data.sort_values(by=['user_id', 'order_id'])
      data = data.fillna(0)  # Fill missing values with 0 for now

      le = LabelEncoder()
      data['encoded_problem_id'] = le.fit_transform(data['problem_id'])

      
      grouped = data.groupby('user_id')

      seq = []
      lab = []
      for user, group in tqdm(grouped):
         group = group.sort_values(by='order_id')
         feature_seq = group['encoded_problem_id'].to_numpy()
         seq.append(torch.tensor(feature_seq, dtype=torch.float32))
         
         blank_labels = np.full((group.shape[1], self.vocab_size), -1)

         # use feature_seq vals as index
         for i in range(len(feature_seq)):
            blank_labels[i][feature_seq[i]] = group['correct'].iloc[i]
         lab.append(torch.tensor(blank_labels, dtype=torch.float32))         
         
         # Break up feature seq into chunks of 100 and pad from there

         # Make large tensor of shape [batch_size,timesteps,vocab_size], full of -1
         # for each timestep, set relevent index to value of group[correct]

      # Padding sequences with -1 using PyTorch's pad_sequence
      padded_seq = pad_sequence(seq, batch_first=True, padding_value=-1)  # (batch_size, timesteps)
      padded_lab = pad_sequence(lab, batch_first=True, padding_value=-1)  # (batch_size, timesteps)

      # Reshape labels to have 3D shape (batch_size, timesteps, 1)
      padded_lab = padded_lab.unsqueeze(-1)

      return padded_seq, padded_lab
   

   # def loss_func(self, y_true, y_pred):



   def fit(self, data):
      print("Beginning data preprocessing")
      X, y = self.preprocess(data)
      print("Data preprocessing finished, beginning fitting.")
      self.model.add(Masking(mask_value=-1, input_shape=(None, 1)))
      self.model.add(Embedding(input_dim=self.vocab_size, output_dim=self.numDim, input_length=None))
      self.model.add(Dropout(0.2))
      self.model.add(LSTM(124, activation='tanh', return_sequences=True))
      self.model.add(Dense(self.vocab_size, activation='sigmoid'))

      self.model.compile(optimizer='adam', 
                  loss=self.loss_func,
                  metrics=['accuracy', 'AUC'])
      
      self.model.fit(X, y, 
         validation_data=(X, y),
         epochs=10, 
         batch_size=64)
      

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



