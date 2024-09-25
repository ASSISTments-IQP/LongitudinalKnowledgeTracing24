import pandas as pd
import scipy.sparse as sp
import numpy as np
import sys
from tqdm import tqdm

print('encoding ' + sys.argv[1])

df = pd.read_csv(sys.argv[1])
df.sort_values(by=['user_xid','skill_id','start_time'], inplace=True)

skills = np.sort(df.skill_id.unique().to_numpy())
n_s = len(skills)

res_arr = sp.dok_array((df.shape[0], 3*n_s))
i = 0
for idx, row in tqdm(df.iterrows()):
    #user_idx = np.where(users == row['user_xid'])[0][0]
    skill_idx = np.where(skills == row['skill_id'])[0][0]
    # res_arr[i,user_idx] = 1
    # res_arr[i,problem_idx] = 1
    res_arr[i,skill_idx] = 1
    res_arr[i,n_s+skill_idx] = row['wins']
    res_arr[i,2*n_s+skill_idx] = row['fails']
    i += 1

res_arr_coo = res_arr.tocoo()
y = df.correct.to_numpy(copy=True)
fn = sys.argv[1].split('.')[0]
fn_X = fn + '_X.npz'
fn_y = fn + '_y.npy'
sp.save_npz(fn_X, res_arr_coo)
np.save(fn_y, y)

