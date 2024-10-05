import numpy as np
import pandas as pd
from pyBKT.models import Model
from tqdm import tqdm
from hmmlearn.hmm import CategoricalHMM
from sklearn.metrics import log_loss, roc_auc_score

class BKTModel:
    def __init__(self, n_iter=10, verbose=True):
        self.n_iter = n_iter
        self.n_s = 0
        self.verbose = verbose
        self.skills = []
        self.model = None
        self.defaults = {'order_id':'order_id','user_id':'user_xid','skill_name':'skill_id','correct':'discrete_score'}


    # Data sequencing (sorting)
    def preprocess(self, data):
        data = data.sort_values(by=['user_xid', 'skill_id', 'start_time'])
        grouped = data.groupby(by=['skill_id', 'user_xid'])
        seq_lens = []
        for value, group in tqdm(grouped, disable=not self.verbose):
            length = len(group)
            seq_lens.append(np.arange(length))

        order_ids = np.concatenate(seq_lens)
        data['order_id'] = order_ids
        return data

    def fit(self, data):

        if self.verbose:
            print("Beginning data preprocessing.")
        data = self.preprocess(data)

        disable = True
        if self.verbose:
            print("Finished data processing. Beginning fitting process.")
        self.model = Model(seed=50,num_fits=5)
        self.model.fit(data=data,defaults=self.defaults)


        auc = self.model.evaluate(data=data,defaults=self.defaults)

        if self.verbose:
            print("Finished model training. Printing final statistics...")
            print(f'Training AUC: {auc}')

        return auc


    def eval(self, data):
        if self.verbose:
            print("Beginning data preprocessing.")
        data = self.preprocess(data)
        auc = self.model.evaluate(data=data,defaults=self.defaults)

        if self.verbose:
            print(f'Eval AUC: {auc}')

        return auc
