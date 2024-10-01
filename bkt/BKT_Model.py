import numpy as np
import pandas as pd
from tqdm import tqdm
from hmmlearn.hmm import CategoricalHMM
from sklearn.metrics import log_loss, roc_auc_score

class BKTModel:
    def __init__(self, n_iter=100, verbose=1):
        self.n_iter = 100
        self.n_s = 0
        self.verbose = verbose
        self.skills = []
        self.models = {}


    # Data sequencing (sorting)
    def preprocess(self, data):
        data = data.sort_values(by=['user_xid', 'skill_id', 'start_time'])
        grouped = data.groupby(by=['skill_id', 'user_xid'])

        curr_skill = 1
        skill_dict = {}
        value_dict = {"seq": [], "lengths": []}

        disable = True
        if self.verbose > 0:
            disable = False
        for value, group in tqdm(grouped, disable=disable):
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

        if self.verbose > 0:
            print("Beginning data preprocessing.")
        skill_dict = self.preprocess(data)

        self.skills = skill_dict.keys()
        self.n_s = len(self.skills)

        lls = []
        aucs = []

        oov_prior = 0
        oov_transmat = np.zeros((2,2))
        oov_emmissionprob = np.zeros((2,2))

        disable = True
        if self.verbose > 0:
            disable = False
            print("Finished data processing. Beginning fitting process.")
        for skill, data in tqdm(skill_dict.items(), disable=disable):
            self.models[skill] = CategoricalHMM(n_components=2, n_iter=self.n_iter, tol=1e-4)

            X = skill_dict[skill]['seq']
            lengths = skill_dict[skill]['lengths']

            if max(lengths) < 5:
                continue

            self.models[skill].fit(X, lengths)

            oov_prior += self.models[skill].startprob_
            oov_transmat += self.models[skill].transmat_
            oov_emmissionprob += self.models[skill].emissionprob_

            state_probs = self.models[skill].predict_proba(X, lengths)
            y_pred = state_probs.dot(self.models[skill].emissionprob_[:,1])
            y_true = np.reshape(X,X.shape[0])

            loss = log_loss(y_true, y_pred, labels=[0,1])  # logistical loss
            auc = roc_auc_score(y_true, y_pred)
            lls.append(loss)
            aucs.append(auc)


        ll = np.mean(lls)
        auc = np.mean(aucs)

        if self.verbose > 0:
            print("Finished model training. Printing final statistics...")
            print(f'Training Log Loss: {ll}')
            print(f'Training AUC: {auc}')

        prior = oov_prior / self.n_s
        trans = oov_transmat / self.n_s
        em = oov_emmissionprob / self.n_s

        oov_mod = CategoricalHMM(n_components=2)
        oov_mod.startprob_ = prior
        oov_mod.transmat_ = trans
        oov_mod.emissionprob_ = em

        self.models[-1] = oov_mod

        return auc


    def eval(self, data):
        if self.verbose > 0:
            print("Beginning data preprocessing.")
        skill_dict = self.preprocess(data)
        lls = []
        aucs = []

        disable = True
        if self.verbose > 0:
            disable = False
        for skill, data in tqdm(skill_dict.items(), disable=disable):
            X = skill_dict[skill]['seq']
            lengths = skill_dict[skill]['lengths']

            if not skill in self.skills:  # Handle OOV skills
                skill = -1

            state_probs = self.models[skill].predict_proba(X, lengths)
            y_pred = state_probs.dot(self.models[skill].emissionprob_[:, 1])
            y_true = np.reshape(X, X.shape[0])

            loss = log_loss(y_true, y_pred)  # logistical loss
            auc = roc_auc_score(y_true, y_pred)
            lls.append(loss)
            aucs.append(auc)

        ll = np.mean(lls)
        auc = np.mean(aucs)

        if self.verbose > 0:
            print(f'Eval Log Loss: {ll}')
            print(f'Eval AUC: {auc}')

        return auc
