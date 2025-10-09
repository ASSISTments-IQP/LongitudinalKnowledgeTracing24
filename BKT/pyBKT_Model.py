import numpy as np
from sklearn.metrics import log_loss, roc_auc_score, f1_score
from Model import Model
from pyBKT.models import Model

class pyBKT_wrapper(Model):
    def __init__(self):
        self.skills = []
        self.pyBKT_Model = None

    def preprocess(self, data, fitting=False):
        if fitting:
            self.skills = data.skill_id.unique()
        else:
            data.loc[~data['skill_id'].isin(self.skills), 'skill_id'] = -1

        self.defaults = {'user_id': 'user_xid', 'skill_name': 'skill_id', 'correct': 'discrete_score', 'forgets': True}
        return data.sort_values(by=['user_xid', 'skill_id', 'start_time'])


    def fit(self, data):
        data = self.preprocess(data, fitting=True)

        self.pyBKT_Model = Model(num_fits=5, parallel=True, defaults=self.defaults)
        self.pyBKT_Model.fit(data=data, forgets=True, defaults = self.defaults)


        #OOV model
        priors = []
        learns = []
        guesses = []
        slips = []
        forgets = []


        for skill_id, param_dict in self.pyBKT_Model.coef_.items():
            priors.append(param_dict['prior'])
            learns.append(param_dict['learns'][0])
            guesses.append(param_dict['guesses'][0])
            slips.append(param_dict['slips'][0])
            forgets.append(param_dict['forgets'][0])

        oov_prior = np.average(priors)
        oov_learn = np.average(learns)
        oov_guesses = np.average(guesses)
        oov_slips = np.average(slips)
        oov_forgets = np.average(forgets)

        cf = self.pyBKT_Model.coef_
        cf[-1] = {
            'prior' : oov_prior,
            'learns' : np.array([oov_learn]),
            'guesses' : np.array([oov_guesses]),
            'slips' : np.array([oov_slips]),
            'forgets' : np.array([oov_forgets])
        }

        self.pyBKT_Model.coef_ = cf

        print(self.pyBKT_Model.evaluate(data=data, metric='auc'))


    def evaluate(self, data):
        data = self.preprocess(data)

        pred_df = self.pyBKT_Model.predict(data=data, defaults=self.defaults)

        y = pred_df['discrete_score']
        y_pred = pred_df['correct_predictions']
        y_pred_classes = np.round(y_pred)

        ll = log_loss(y, y_pred)
        auc = roc_auc_score(y, y_pred)
        f1 = f1_score(y, y_pred_classes)

        return auc, ll, f1

