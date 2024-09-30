import numpy as np
import pandas as pd
from hmmlearn.hmm import CategoricalHMM
from sklearn.metrics import log_loss, roc_auc_score

class BKTModel:
    def __init__(self):
        self.model = CategoricalHMM(n_components=2, init_params="", n_iter=100)

        start_prob = np.array([0.5, 0.5])
        trans_prob = np.array([[0.7, 0.3],[0.3, 0.7]])
        emission_prob = np.array([[0.8, 0.2],[0.2, 0.8]])
        self.model.startprob_ = start_prob
        self.model.transmat_ = trans_prob
        self.model.emission_prob_ = emission_prob

    # Data sequencing (sorting)
    def preprocess(self, data):
        data.sort_values(by=['user_xid','skill_id','start_time'], inplace=True)
        gk = data.groupby(by=['user_xid', 'skill_id'])['discrete_score'].apply(list)
        X = np.concatenate(gk.values).reshape(-1, 1)  # reshape into array
        lens = [len(seq) for seq in gk]               # the length of each sequence in our grouped dataframe
        return X, lens

    def fit(self, data):

        print("Beginning data preprocessing.")
        X, lens = self.preprocess(data)
        print("Finished data processing. Beginning fitting process.")

        self.model.fit(X, lens)

        print("Finished model training. Printing final statistics...")
        y_score = self.model.predict_proba(X)[:, 1]

        loss = log_loss(X, y_score)     # logistical loss
        auc = roc_auc_score(X, y_score)

        print(f"Loss: {loss}")
        print(f"AUC:  {auc}")

        return loss, auc