# from sakt.SAKT_model import SAKTModel
from DKT.DKT_pt import DKT
from sakt.sakt_pt import SAKTModel
from tqdm import tqdm
import pandas as pd
import sys, json
model_list = ['DKT','SAKT-E','SAKT-KC']


def run_one_fold(val_fold, data, model_type, year):
    train = data.pop(val_fold)
    validation = pd.concat(data)

    if model_type == 'DKT':
        model = DKT(32, 40, 256, 3e-2)
        num_epochs = 50
    elif model_type == 'SAKT-E':
        model = SAKTModel(70,64,288,8,0.14,4e-4,0.95,feature_col='old_problem_id')
        num_epochs = 25
    else:
        model = SAKTModel(70,64,288,8,0.14,4e-4,0.95,feature_col='skill_id')
        num_epochs = 25

    model.fit(train, num_epochs=num_epochs)
    print(f"{model_type} fit for {year} with hold-out fold {val_fold}")
    return model.evaluate(validation)


if __name__ == '__main__':
    year_list = ['19-20','20-21','21-22','22-23','23-24']

    if len(sys.argv) != 4:
        raise Exception("Invalid number of args specified")

    model_type = sys.argv[1]
    year = sys.argv[2]
    holdout_fold_num = int(sys.argv[3])

    if model_type not in model_list:
        raise Exception(f"Invalid model type specified, model type must be one of {str(model_list)[1:-1]}")
    elif year not in year_list:
        raise Exception(f'Invalid year specified, model year must be one of {str(year_list)[1:-1]}')
    elif holdout_fold_num not in range(1,11):
        raise Exception(f"Invalid fold number provided")

    sample_dict = {}

    print('Loading year samples')
    for y in tqdm(year_list):
        y_dict = {}
        for i in range(1, 11):
            s1 = pd.read_csv(f'../Data/samples/{y}/sample{i}.csv')
            y_dict[i] = s1

        sample_dict[y] = y_dict

    print('Samples loaded & processed into folds')

    res_tup = run_one_fold(holdout_fold_num, sample_dict[year], model_type, year)

    res = {
        year: {
            holdout_fold_num: {
                'auc':res_tup[0],
                'll':res_tup[1],
                'f1':res_tup[2]
            }}
    }

    with open(f'./wy_DKT_{year}_{holdout_fold_num}.json','w') as fout:
        json.dump(res,fout)
        fout.close()
