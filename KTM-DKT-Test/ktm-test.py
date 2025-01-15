import numpy as np
import pandas as pd
from EduKTM import DKT
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

class DKT_KTM():
    def __init__(self, batch_size=64, num_steps=50, hidden_size=128, num_layers=1, lr=1e-4):
        self.vocab = []
        self.vocab_size = 0
        self.enc_dict = {}
        self.num_steps = num_steps
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.batch_size = batch_size
        self.model = None

    def preprocess(self, df, fitting=False):
        if fitting:
            self.vocab = df['skill_id'].unique().tolist()
            self.vocab_size = len(self.vocab) + 1
            self.enc_dict = {sk_id: i for i, sk_id in enumerate(self.vocab, start=1)}

        df.drop_duplicates('problem_log_id', inplace=True)
        df.sort_values(by=['user_xid','start_time'], inplace=True)

        seqs = []
        for name, group in tqdm(df.groupby(by='user_xid')):
            group_len = group.shape[0]
            mod = 0 if group_len % self.num_steps == 0 else (self.num_steps - group_len % self.num_steps)
            oh = np.zeros(shape=(group_len + mod, self.vocab_size * 2))
            i = 0
            for idx, row in group.iterrows():
                skill = row['skill_id']
                corr = row['discrete_score']
                found_vocab = self.check_vocab(skill)
                col_idx = found_vocab if corr == 0 else found_vocab + self.vocab_size
                oh[i][col_idx] = 1
                i += 1
            seqs.append(oh)

        seqs = np.concatenate(seqs)
        full_data = torch.FloatTensor(seqs.reshape(-1, self.num_steps, 2 * self.vocab_size))
        d_l = DataLoader(full_data, batch_size=self.batch_size)
        return d_l


    def fit(self, train_data, num_epochs=3):
        raw_q_array = self.preprocess(train_data, True)
        self.model = DKT(self.vocab_size, self.hidden_size, self.num_layers)
        self.model.train(raw_q_array, epoch=num_epochs, lr=self.lr)

        return self.model.eval(raw_q_array)

    def evaluate(self, test_data):
        raw_q_array = self.preprocess(test_data)
        return self.model.eval(raw_q_array)

    def check_vocab(self, key):
        return self.enc_dict.get(key, 0)


if __name__ == '__main__':
    print(torch.cuda.is_available())
    torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train = pd.read_csv('../Data/samples/23-24/sample1.csv')
    test = pd.read_csv('../Data/samples/23-24/sample2.csv')

    model = DKT_KTM(num_steps=50)
    print('Fitting now')
    print(model.fit(train))
    print('Training finished. eval')
    print(model.evaluate(test))
