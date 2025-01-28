from PFA.PFA_Model import PFA
from bkt.BKT_Model import BKTModel
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import sys, json
model_list = ['BKT','PFA']


def run_cv_one_fold(data, test_fold_num, model_type):
    train = data.pop(test_fold_num)
    test = pd.concat(data.values())

    if model_type == 'BKT':
        model = BKTModel()
    if model_type == 'PFA':
        model = PFA()


    model.fit(train)
    eval_tup = model.evaluate(test)
    res = {
        'auc': eval_tup[0],
        'll': eval_tup[1],
        'f1': eval_tup[2]
    }

    return res, test_fold_num


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
    for i in range(1, 11):
        s1 = pd.read_csv(f'../Data/samples/{train_year}/sample{i}.csv')

        fold_dict[i] = s1

    print('Samples loaded')

    res = {}
    args = zip([fold_dict] * 10, range(1,11), [model_type] * 10)
    with Pool(10) as p:
        for l in p.starmap(run_cv_one_fold, args):
            res[l[1]] = l[0]

    with open(f'./within_year_results_{model_type}_{train_year}.json','w') as fout:
        json.dump(res,fout)
        fout.close()
