from sakt_pt import SAKTModel
import pandas as pd
import numpy as np


train_data = pd.read_csv('../Data/samples/19-20/sample1.csv')



mod = SAKTModel(num_steps=50, batch_size = 32, d_model=128, num_heads=8, dropout_rate=0.2, init_learning_rate=1e-3, learning_decay_rate=0.98)
mod.fit(train_data, num_epochs=5)
