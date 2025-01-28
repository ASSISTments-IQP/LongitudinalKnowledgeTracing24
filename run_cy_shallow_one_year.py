from PFA.PFA_Model import PFA
from bkt.BKT_Model import BKTModel
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import sys, json
model_list = ['BKT','PFA']


def run_one_sample(model_type, train_samples, test_samples, sample_num):
    train = train_samples[sample_num]
    tests = {}
    for year, samps in test_samples.items():
        tests[year] = (samps[sample_num])

    if model_type == 'BKT':
        model = BKTModel()
    if model_type == 'PFA':
        model = PFA()

    model.fit(train)

    res = {}
    for year, samp in tests.items():
        eval_tup = model.evaluate(samp)
        res[year] = {
            'auc': eval_tup[0],
            'll': eval_tup[1],
            'f1': eval_tup[2]
        }

    return res, sample_num



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

    train_dict = {}
    for i in range(1, 11):
        s1 = pd.read_csv(f'../Data/samples/{train_year}/sample{i}.csv')
        train_dict[i] = s1

    test_years = year_list[year_list.index(train_year):]

    print('Loading year samples')
    test_dict = {}
    for y in tqdm(test_years):
        y_dict = {}
        for i in range(1,11):
            s1 = pd.read_csv(f'../Data/samples/{y}/sample{i}.csv')
            y_dict[i] = s1

        test_dict[y] = y_dict


    res = {}
    args = zip([model_type] * 10, [train_dict] * 10, [test_dict] * 10, range(1,11))
    with Pool(10) as p:
        for l in p.starmap(run_one_sample, args):
            res[l[1]] = l[0]

    with open(f'./cross_year_results_{model_type}_{train_year}.json','w') as fout:
        json.dump(res,fout)
        fout.close()

