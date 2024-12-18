from DKT.DKT_pt import DKT
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import sys, json
model_list = ['DKT-KC']


def run_one_sample(model_type, train_samples, test_samples, sample_num):
    train = train_samples[sample_num]
    tests = {}
    for year, samps in test_samples.items():
        tests[year] = (samps[sample_num])

    gpu_num = sample_num - 1

    needs_num_epochs = True

    
    if model_type == 'DKT-KC':
        num_epochs = 3
        model = DKT(16,50,128,0.33,1e-4,gpu_num=gpu_num,feature_col='skill_id')
    

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
    
    model_type = 'DKT-KC'
    train_year = '19-20'

    if model_type not in model_list:
        raise Exception(f"Invalid model type specified, model type must be one of {str(model_list)[1:-1]}")
    elif train_year not in year_list:
        raise Exception(f'Invalid year specified, model year must be one of {str(year_list)[1:-1]}')

    train_dict = {}
    s1 = pd.read_csv(f'samples/{train_year}/sample{1}.csv')
    train_dict[0] = s1

    test_years = year_list[year_list.index(train_year):]

    print('Loading year samples')
    test_dict = {}
    for y in tqdm(test_years):
        y_dict = {}
        s1 = pd.read_csv(f'samples/{y}/sample{1}.csv')
        y_dict[0] = s1

        test_dict[y] = y_dict


    res = {}
    args = zip([model_type] * 1, [train_dict] * 1, [test_dict] * 1, range(1))
    with Pool(1) as p:
        for l in p.starmap(run_one_sample, args):
            res[l[1]] = l[0]

    with open(f'./cross_year_results_{model_type}_{train_year}.json','w') as fout:
        json.dump(res,fout)
        fout.close()

