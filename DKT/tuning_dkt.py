import optuna
from DKT_pt import DKT
import pandas as pd
import numpy as np
import random
from multiprocessing import Pool


def run_one_fold(train_data, test_data, ns, bs, dm, lr, ne, dr, rl):
    print((ns,bs,dm,lr,ne))

    train_data.drop_duplicates(subset=['problem_log_id'])
    test_data.drop_duplicates(subset=['problem_log_id'])

    train_data.sort_values(by=['user_xid', 'start_time'], inplace=True)
    test_data.sort_values(by=['user_xid', 'start_time'], inplace=True)

    mod = DKT(bs, ns, dm, lr, dr, rl)
    mod.fit(train_data, num_epochs=ne)
    return mod.evaluate(test_data)[0]


def objective(trial):
    df = pd.read_csv('../Data/samples/validation_sample.csv')

    alogs = df.assignment_log_id.unique()
    np.random.shuffle(alogs)
    folds = np.array_split(alogs, 2)
    train = df[df['assignment_log_id'].isin(folds[0])].copy()
    test = df[df['assignment_log_id'].isin(folds[1])].copy()

    num_steps = trial.suggest_int('num_steps', 20, 100, step = 10)
    batch_size = trial.suggest_int('batch_size', 16, 64, step=8)
    d_model = trial.suggest_int('d_model', 64, 512, step = 32)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    reg_lambda = trial.suggest_float('reg_lambda', 1e-6, 1e-2, log=True)
    num_epochs = trial.suggest_int('num_epochs', 100, 200)

    return run_one_fold(train, test, num_steps, batch_size, d_model, learning_rate, num_epochs, dropout_rate, reg_lambda)


if __name__ == '__main__':
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=50)


    print("Best hyperparameters:", study.best_params)
    print("Best validation ACC:", study.best_value)
    
    
