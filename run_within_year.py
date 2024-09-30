from PFA.PFA_Model import PFA
from sakt.SAKT_model import SAKTModel
from concurrent.futures import ProcessPoolExecutor as Pool
from functools import partial
from tqdm import tqdm
import pandas as pd
import sys
model_list = ['BKT','PFA','DKT','SAKT']


def run_cv(items, model_type):
    year = items[0]
    data = items[1]

    res_l = []

    print(f'Running CV for {year}')
    with Pool() as p:
        for k in p.map(partial(run_one_fold,data=data, model_type=model_type, year=year), range(5)):
            res_l.append(k)

    return res_l


def run_one_fold(val_fold, data, model_type, year):
    train_list = []
    for key, val in data.items():
        if key == val_fold:
            validation = val
        else:
            train_list.append(val)

    train = pd.concat(train_list)

    if model_type == 'PFA':
        model = PFA(verbose=1)
    if model_type == 'SAKT':
        model = SAKTModel()

    model.fit(train)
    print(f"{model_type} fit for {year} with hold-out fold {val_fold}")
    return model.eval(validation)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise Exception("No model specified. Please specify which model you would like to run the within-year analysis")
    elif sys.argv[1] not in model_list:
        raise Exception(f"Invalid model type specified, model type must be one of {str(model_list)[1:-1]}")

    year_list = ['19-20','20-21','21-22','22-23','23-24']
    sample_dict = {}

    print('Loading year samples')
    for y in tqdm(year_list):
        y_dict = {}
        j = 0
        for i in range(1,11,2):
            s1 = pd.read_csv(f'./Data/samples/{y}/sample{i}.csv')
            s2 = pd.read_csv(f'./Data/samples/{y}/sample{i+1}.csv')

            y_dict[j] = pd.concat([s1,s2], ignore_index=True)
            j += 1

        sample_dict[y] = y_dict

    print('Samples loaded & processed into folds')

    res = []

    with Pool() as p:
        for l in p.map(partial(run_cv, model_type = sys.argv[1]) , sample_dict.items()):
            res.append(l)

    print(res)
