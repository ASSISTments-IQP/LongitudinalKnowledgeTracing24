import optuna
import pandas as pd
import numpy as np
import random, sys


def run_one_fold(train_data, test_data, ns, bs, dm, nh, dr, ne, ilr, ldr):
    print((ns,bs,dm,nh,dr,ne,ilr,ldr))
    from sakt_pt import SAKTModel
    if model_type == 'E':
        f_col = 'old_problem_id'
    elif model_type == 'KC':
        f_col = 'skill_id'

    train_data.drop_duplicates(subset=['problem_log_id'])
    test_data.drop_duplicates(subset=['problem_log_id'])

    train_data.sort_values(by=['user_xid', 'start_time'], inplace=True)
    test_data.sort_values(by=['user_xid', 'start_time'], inplace=True)

    mod = SAKTModel(ns, bs, dm, nh, dr, ilr, ldr, feature_col=f_col)
    mod.fit(train_data, num_epochs=1)
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
    num_heads = trial.suggest_categorical('num_heads', [2,4,8,16,32])
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    num_epochs = trial.suggest_int('num_epochs', 10, 40)
    init_learning_rate = trial.suggest_float('init_learning_rate', 1e-4, 1e-2, log=True)
    learning_decay_rate = trial.suggest_float('learning_decay_rate', 0.7, 0.99)

    return run_one_fold(train, test, num_steps, batch_size, d_model, num_heads, dropout_rate, num_epochs, init_learning_rate, learning_decay_rate)


if __name__ == '__main__':
    model_type = sys.argv[1]

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=50)

    print("Best hyperparameters:", study.best_params)
    print("Best validation AUC:", study.best_value)
    
    
