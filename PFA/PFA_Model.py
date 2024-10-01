import scipy.sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
import scipy.sparse as sp
import numpy as np
from tqdm import tqdm


class PFA:
    def __init__(self, verbose):
        self.skills = []
        self.n_s = len(self.skills)
        self.verbose = verbose
        self.model = LogisticRegression(penalty=None, solver='lbfgs', verbose=verbose, max_iter=10 ** 3)


    def preprocess(self, data):
        # Create wins & fails aggregators & make sure data is sorted correctly
        wins = []
        fails = []
        data.sort_values(by=['user_xid','skill_id','start_time'], inplace=True)

        # Loop (w/ tqdm if verbose) to create wins & fails features
        if self.verbose > 0:
            for name, group in tqdm(data.groupby(by=['user_xid', 'skill_id'])):
                w = 0
                f = 0
                for idx, row in group.iterrows():
                    wins.append(w)
                    fails.append(f)
                    if row['discrete_score'] == 1:
                        w += 1
                    else:
                        f += 1
        else:
            for name, group in data.groupby(by=['user_xid', 'skill_id']):
                w = 0
                f = 0
                for idx, row in group.iterrows():
                    wins.append(w)
                    fails.append(f)
                    if row['discrete_score'] == 1:
                        w += 1
                    else:
                        f += 1

        data['wins'] = wins
        data['fails'] = fails


        # Start creating the sparse array for the data
        skill_arr = sp.dok_array((data.shape[0], self.n_s+1))
        win_arr = sp.dok_array((data.shape[0], self.n_s+1))
        fail_arr = sp.dok_array((data.shape[0], self.n_s+1))
        i = 0

        if self.verbose > 0:
            for idx, row in tqdm(data.iterrows()):
                if row['skill_id'] in self.skills:
                    skill_idx = np.where(self.skills == row['skill_id'])[0][0]
                else:
                    skill_idx = self.n_s

                skill_arr[i,skill_idx] = 1
                win_arr[i,skill_idx] = row['wins']
                fail_arr[i,skill_idx] = row['fails']
                i += 1
        else:
            for idx, row in data.iterrows():
                if row['skill_id'] in self.skills:
                    skill_idx = np.where(self.skills == row['skill_id'])[0][0]
                else:
                    skill_idx = self.n_s

                skill_arr[i, skill_idx] = 1
                win_arr[i, skill_idx] = row['wins']
                fail_arr[i, skill_idx] = row['fails']
                i += 1

        res_arr = sp.hstack([skill_arr,win_arr,fail_arr])

        X = res_arr.tocoo()
        y = data.discrete_score.to_numpy(copy=True)

        return X, y


    def fit(self, data):
        self.skills = np.sort(data.skill_id.unique())
        self.n_s = len(self.skills)

        print("Beginning data preprocessing")
        X, y = self.preprocess(data)
        print("Data preprocessing finished, beginning fitting.")

        self.model.fit(X,y)

        print("Model Training finished, final statistics:")
        y_pred = self.model.predict_proba(X)[:,1]

        ll = log_loss(y,y_pred)
        auc = roc_auc_score(y,y_pred)

        print(f"Training loss: {ll}")
        print(f"Training AUC: {auc}")

        return auc


    def eval(self, data):
        if self.n_s == 0:
            print("No model has been trained, aborting")
            return

        print("Beginning data preprocessing")
        X, y = self.preprocess(data)

        y_pred = self.model.predict_proba(X)[:, 1]

        ll = log_loss(y, y_pred)
        auc = roc_auc_score(y, y_pred)

        print(f"Log loss: {ll}")
        print(f"AUC: {auc}")

        return auc
