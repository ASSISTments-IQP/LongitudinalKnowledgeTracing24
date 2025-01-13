from PFA.PFA_Model import PFA
from bkt.BKT_Model import BKTModel
from sakt.sakt_pt import SAKTModel
from DKT.DKT_pt import DKT
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import sys, json
model_list = ['BKT','PFA','DKT-E','DKT-KC','SAKT-E','SAKT-KC']


def run_one_sample(model_type, train_samples, test_samples, sample_num):
    train = train_samples[sample_num]
    tests = {}
    for year, samps in test_samples.items():
        tests[year] = (samps[sample_num])

    gpu_num = sample_num - 1

    needs_num_epochs = True

    if model_type == 'BKT':
        needs_num_epochs = False
        model = BKTModel()
    if model_type == 'PFA':
        needs_num_epochs = False
        model = PFA()
    # if model_type == 'DKT-E':
    #     num_epochs = 3
    #     model = DKT_model(gpu_num=gpu_num, feature_col='old_problem_id')  # UPDATE HYPERPARAMS LATER
    if model_type == 'DKT-KC':
        num_epochs = 3
        model = DKT(40, 80, 448, 0.1, 1e-3, 1e-4, gpu_num=gpu_num, feature_col='skill_id')
    if model_type == 'SAKT-E':
        num_epochs = 6
        model = SAKTModel(70,64,288,8,0.14,4e-4,0.95,gpu_num=gpu_num,feature_col='old_problem_id')  # UPDATE HYPERPARAMS LATER
    if model_type == 'SAKT-KC':
        num_epochs = 6
        model = SAKTModel(70,64,288,8,0.14,4e-4,0.95,gpu_num=gpu_num,feature_col='skill_id')

    if needs_num_epochs:
        model.fit(train, num_epochs)
    else:
        model.fit(train)

    res = {}
    for year, samp in tests.items():
        res[year] = model.evaluate(samp)

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

