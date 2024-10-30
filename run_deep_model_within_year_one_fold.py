from sakt.SAKT_model import SAKTModel
from DKT.DKT_Model import DKT
from tqdm import tqdm
import pandas as pd
import sys, json
model_list = ['DKT','SAKT-E','SAKT-KC']


def run_one_fold(val_fold, data, model_type, year):
    train_list = []
    for key, val in data.items():
        if key == val_fold:
            validation = val
        else:
            train_list.append(val)

    train = pd.concat(train_list)

    if model_type == 'SAKT-E':
        model = SAKTModel(problem=True)
    if model_type == 'SAKT-KC':
        model = SAKTModel(problem=False)
    if model_type == 'DKT':
        model = DKT()

    model.fit(train, num_epochs=1)
    print(f"{model_type} fit for {year} with hold-out fold {val_fold}")
    return model.eval(validation)


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
    elif holdout_fold_num not in range(5):
        raise Exception(f"Invalid fold number provided")

    sample_dict = {}

    print('Loading year samples')
    for y in tqdm(year_list):
        y_dict = {}
        j = 0
        for i in range(1,11,2):
            s1 = pd.read_csv(f'../Data/samples/{y}/sample{i}.csv')
            s2 = pd.read_csv(f'../Data/samples/{y}/sample{i+1}.csv')

            y_dict[j] = pd.concat([s1,s2], ignore_index=True)
            j += 1

        sample_dict[y] = y_dict

    print('Samples loaded & processed into folds')


    res = {
        year: {holdout_fold_num: run_one_fold(holdout_fold_num, sample_dict[year], model_type, year)}
    }

    with open(f'./wy_{model_type}_{year}_{holdout_fold_num}.json','w') as fout:
        json.dump(res,fout)
        fout.close()
