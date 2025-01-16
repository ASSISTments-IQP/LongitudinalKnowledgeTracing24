import optuna
from DKT_pt import DKT
import pandas as pd
import numpy as np
from multiprocessing import Pool


def run_one_fold(data, test_fold, ns, bs, dm, lr, ne):
    print((ns,bs,dm,lr,rl,ne))
    test_data = data.pop(test_fold)
    train_data = pd.concat(data)

    train_data.drop_duplicates(subset=['problem_log_id'])
    test_data.drop_duplicates(subset=['problem_log_id'])

    train_data.sort_values(by=['user_xid', 'start_time'], inplace=True)
    test_data.sort_values(by=['user_xid', 'start_time'], inplace=True)

    mod = DKT(bs, ns, dm, lr, gpu_num=test_fold)
    mod.fit(train_data, num_epochs=ne)
    return mod.evaluate(test_data)


def objective(trial):
    df = pd.read_csv('../Data/samples/validation_sample.csv')

    alogs = df.assignment_log_id.unique()
    np.random.shuffle(alogs)
    folds = np.array_split(alogs, 4)

    data = []
    for i in range(4):
        data.append(df[df['assignment_log_id'].isin(folds[i])].copy())

    num_steps = trial.suggest_int('num_steps', 20, 50, step = 10)
    batch_size = trial.suggest_int('batch_size', 16, 32, step=8)
    d_model = trial.suggest_int('d_model', 64, 256, step = 16)
    # dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-1, log=True)
    # reg_lambda = trial.suggest_float('reg_lambda', 1e-4, 1, log=True)
    num_epochs = trial.suggest_int('num_epochs', 1, 10)

    res = []
    args = zip([data] * 4, range(4), [num_steps] * 4, [batch_size] * 4, [d_model] * 4, [learning_rate] * 4, [num_epochs] * 4)
    with Pool(4) as p:
        for l in p.starmap(run_one_fold, args):
            res.append(l)

    return np.mean(res)


if __name__ == '__main__':
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=50)


    print("Best hyperparameters:", study.best_params)
    print("Best validation ACC:", study.best_value)
    
    
