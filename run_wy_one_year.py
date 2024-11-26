from PFA.PFA_Model import PFA
from bkt.BKT_Model import BKTModel
from sakt.SAKT_model import SAKTModel
from DKT.DKT_pt import DKT
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import sys, json
model_list = ['BKT','PFA','DKT-E','DKT-KC','SAKT-E','SAKT-KC']


def run_cv_one_fold(data, test_fold_num, model_type):
    test = data.pop(test_fold_num)
    train = pd.concat(data.values())
    needs_num_epochs = True

    if model_type == 'BKT':
        needs_num_epochs = False
        model = BKTModel()
    if model_type == 'PFA':
        needs_num_epochs = False
        model = PFA()
    if model_type == 'DKT-E':
        num_epochs = 3
        model = DKT(16,50,128,0.33,1e-4,test_fold_num,'old_problem_id')  # UPDATE HYPERPARAMS LATER
    if model_type == 'DKT-KC':
        num_epochs = 3
        model = DKT(16,50,128,0.33,1e-4,test_fold_num,'skill_id')
    if model_type == 'SAKT-E':
        pass
       model = SAKTModel()  # UPDATE HYPERPARAMS LATER
    if model_type == 'SAKT-KC':
       model = SAKTModel()
        pass
 
    if needs_num_epochs:
        model.fit(train, num_epochs)
    else:
        model.fit(train)

    return model.eval(test), test_fold_num


if __name__ == '__main__':
    year_list = ['19-20', '20-21', '21-22', '22-23', '23-24']
    if len(sys.argv) != 3:
        raise Exception("Invalid number of args specified")

    model_type = sys.argv[1]
    train_year = sys.argv[2]

    if model_type not in model_list:
        raise Exception(f"Invalid model type specified, model type must be one of {str(model_list)[1:-1]}")
    elif train_year not in year_list:
        raise Exception(f'Invalid year specified, model year must be one of {str(year_list)[1:-1]}')
    sample_dict = {}

    print(f'Loading year samples for year {train_year}')
    fold_dict = {}
    j = 0
    for i in range(1, 11, 2):
        s1 = pd.read_csv(f'../Data/samples/{train_year}/sample{i}.csv')
        s2 = pd.read_csv(f'../Data/samples/{train_year}/sample{i + 1}.csv')

        fold_dict[j] = pd.concat([s1, s2], ignore_index=True)
        j += 1

    print('Samples loaded & processed into folds')

    res = {}
    args = zip([fold_dict] * 5, range(5), [model_type] * 5)
    with Pool(5) as p:
        for l in p.starmap(run_cv_one_fold, args):
            res[l[1]] = l[0]

    with open(f'./within_year_results_{model_type}_{train_year}.json','w') as fout:
        json.dump(res,fout)
        fout.close()
