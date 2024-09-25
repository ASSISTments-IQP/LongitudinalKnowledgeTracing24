from PFA.PFA_Model import PFA
from concurrent.futures import ProcessPoolExecutor as Pool
from functools import partial
from tqdm import tqdm
import pandas as pd
import sys
model_list = ['BKT','PFA','DKT','SAKT']


def run_cv():


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise Exception("No model specified. Please specify which model you would like to run the within-year analysis")
    elif sys.argv[1] not in model_list:
        raise Exception(f"Invalid model type specified, model type must be one of {str(model_list)[1:-1]}")

    year_list = ['19-20','20-21','21-22','22-23','23-24']
    sample_dict = {}

    print('Loading year samples')
    for y in tqdm(year_list):
        y_dict = {}
        j = 0
        for i in range(1,11,2):
            s1 = pd.read_csv(f'./Data/samples/{y}/sample{i}')
            s2 = pd.read_csv(f'./Data/samples/{y}/sample{i+1}')

            y_dict[j] = pd.concat([s1,s2], ignore_index=True)
            j += 1

        sample_dict[y] = y_dict

    print('Samples loaded & processed into folds')

