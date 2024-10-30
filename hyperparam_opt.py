import pandas as pd
import numpy as np
import sys
from multiprocessing import Pool

model_list = ['DKT', 'SAKT-E', 'SAKT-KC']

val_logs = pd.read_csv('./Data/samples/validation_sample.csv')
assignment_log_ids = val_logs.assignment_log_ids.unique()
np.random.shuffle(assignment_log_ids)
fold_assignment_logs = np.array_split(assignment_log_ids, 5)
fold_dict = {}
for i in range(5):
    fold_dict[i] = val_logs[val_logs['assignment_log_id'].isin(fold_assignment_logs[i])].copy()

def obj_SAKT(params):
    d_model, num_heads, dropout_rate, init_learning_rate = params['d_model'], params['num_heads'], params['dropout_rate'], params['init_learning_rate']

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise Exception("Invalid number of args specified")

    model_type = sys.argv[1]

    if model_type not in model_list:
        raise Exception(f"Invalid model type specified, model type must be one of {str(model_list)[1:-1]}")
