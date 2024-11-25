import optuna
from SAKT_model import SAKTModel
import pandas as pd
import numpy as np
from multiprocessing import Process, Queue


def run_one_fold(data, test_fold, ns, bs, dm, nh, dr, ne, ilr, ldr, res_queue):
    test_data = data.pop(test_fold)
    train_data = pd.concat(data)

    train_data.drop_duplicates(subset=['problem_log_id'])
    test_data.drop_duplicates(subset=['problem_log_id'])

    train_data.sort_values(by=['user_xid', 'start_time'], inplace=True)
    test_data.sort_values(by=['user_xid', 'start_time'], inplace=True)

    mod = SAKTModel(ns, bs, dm, nh, dr, ilr, ldr, gpu_num=test_fold)
    mod.fit(train_data, num_epochs=ne)
    res_queue.put(mod.eval(test_data))


def objective(trial):
    df = pd.read_csv('../Data/samples/validation_sample.csv')

    alogs = df.assignment_log_id.unique()
    np.random.shuffle(alogs)
    folds = np.array_split(alogs, 4)

    data = []
    for i in range(4):
        data.append(df[df['assignment_log_id'].isin(folds[i])].copy())

    num_steps = trial.suggest_int('num_steps', 20, 100, step = 10)
    batch_size = trial.suggest_int('batch_size', 16, 64, step=8)
    d_model = trial.suggest_int('d_model', 64, 512, step = 32)
    num_heads = trial.suggest_int('num_heads', 2, 32, step = 2)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    num_epochs = trial.suggest_int('num_epochs', 3, 30)
    init_learning_rate = trial.suggest_float('init_learning_rate', 1e-6, 1e-2, log=True)
    learning_decay_rate = trial.suggest_float('learning_decay_rate', 0.7, 0.99)

    print(batch_size)
    res_queue = Queue()
    procs = []
    for i in range(4):
        p = Process(target=run_one_fold, args=(data.copy(), num_steps, i, batch_size, d_model, num_heads, dropout_rate, num_epochs, init_learning_rate, learning_decay_rate, res_queue,))
        p.start()
        procs.append(p)

    for i in range(4):
        procs[i].join()

    res = []
    for i in range(4):
        res.append(res_queue.get())

    return np.mean(res)


if __name__ == '__main__':
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=100)


    print("Best hyperparameters:", study.best_params)
    print("Best validation AUC:", study.best_value)
    
    
