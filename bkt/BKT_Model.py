import numpy as np
import pandas as pd
from tqdm import tqdm
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
        data = data.sort_values(by=['user_xid', 'skill_id', 'start_time'])
        grouped = data.groupby(by=['skill_id', 'user_xid'])

        curr_skill = 1
        skill_dict = {}
        value_dict = {"seq": [], "lengths": []}

        for value, group in tqdm(grouped):
            if value[0] != curr_skill:
                value_dict['seq'] = np.concatenate(value_dict['seq']).reshape(-1, 1)
                skill_dict[curr_skill] = value_dict
                value_dict = {"seq": [], "lengths": []}

            curr_skill = value[0]
            value_dict['seq'].append(group.discrete_score.to_numpy())
            value_dict['lengths'].append(group.shape[0])

        value_dict['seq'] = np.concatenate(value_dict['seq']).reshape(-1, 1).astype(int)
        skill_dict[curr_skill] = value_dict

        return skill_dict

    def fit(self, data):

        print("Beginning data preprocessing.")
        skill_dict = self.preprocess(data)
        print("Finished data processing. Beginning fitting process.")

        for skill in skill_dict:
            X = skill_dict[skill]['seq']
            lengths = skill_dict[skill]['lengths']

            self.model.fit(X, lengths)

            print("Finished model training. Printing final statistics...")
            trained_start_prob = self.model.startprob_
            trained_trans_prob = self.model.transmat_
            trained_emission_prob = self.model.emissionprob_

            print('Trained Start_Probabilities: ', trained_start_prob)
            print('Trained Transition Probabilities: ', trained_trans_prob)
            print('Trained Emission Probabilities: ', trained_emission_prob)

            y_score = self.model.predict_proba(X)[:, 1]

            loss = log_loss(X, y_score)  # logistical loss
            auc = roc_auc_score(X, y_score)

            print(f"Loss: {loss}")
            print(f"AUC:  {auc}")