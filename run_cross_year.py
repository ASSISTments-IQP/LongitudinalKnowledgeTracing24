from PFA.PFA_Model import PFA
from bkt.BKT_model_pybkt import BKTModel
from sakt.SAKT_model import SAKTModel
from DKT.DKT_Model import DKT
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import sys, json
model_list = ['BKT','PFA','DKT','SAKT']


def run_one_sample(train, test, sample_num, model_type):
    if model_type == 'PFA':
        model = PFA(verbose=0)
    if model_type == 'SAKT':
        model = SAKTModel()
    if model_type == 'BKT':
        model = BKTModel(verbose=0)
    if model_type == 'DKT':
        model = DKT(verbose=0)
    model.fit(train)
    res = model.eval(test)
    return res, sample_num



if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise Exception("No model specified. Please specify which model you would like to run the within-year analysis")
    elif sys.argv[1] not in model_list:
        raise Exception(f"Invalid model type specified, model type must be one of {str(model_list)[1:-1]}")

    years = ['19-20', '20-21', '21-22', '22-23', '23-24']
    sample_dict = {}
    print('Loading year samples')
    for y in tqdm(years):
        y_dict = {}
        for i in range(1,11):
            s1 = pd.read_csv(f'./Data/samples/{y}/sample{i}.csv')
            y_dict[i] = s1

        sample_dict[y] = y_dict

    train_years = years[0:4]
    test_years = years[1:]

    running_years = []
    res = {}

    for train_y in train_years:
        res[train_y] = {}
        for test_y in test_years:
            res[train_y][test_y] = {}
            running_years.append((train_y,test_y))
        test_years.pop(0)


    print(running_years)

    for train_y, test_y in tqdm(running_years):
        args = zip(sample_dict[train_y].values(), sample_dict[test_y].values(), range(1,11), [sys.argv[1]] * 10)
        with Pool(10) as p:
            for l in p.starmap(run_one_sample, args):
                res[train_y][test_y][l[1]] = l[0]
            p.close()
            p.join()

    with open(f'./cross_year_results_{sys.argv[1]}.json','w') as fout:
        json.dump(res,fout)
        fout.close()
