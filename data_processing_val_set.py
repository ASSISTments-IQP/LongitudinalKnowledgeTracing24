#%%
import pandas as pd
from tqdm import tqdm
import os
#%% md
# # Split dataset into different dataframes for each year
#%%
def read_year(year):
    res = []

    with pd.read_csv("all_years_problem_logs.csv",chunksize=10**6) as read:
        for chunk in tqdm(read):
            res.append(chunk[chunk['academic_year'] == year])
            
    return pd.concat(res)
#%%
if os.path.isfile('./Data/full_year/19-20_logs.csv'):
    ay19_20 = pd.read_csv('./Data/full_year/19-20_logs.csv')
else:
    ay19_20 = read_year('19-20')
    ay19_20.to_csv('./Data/full_year/19-20_logs.csv',index=False)
#%%
if os.path.isfile('./Data/full_year/20-21_logs.csv'):
    ay20_21 = pd.read_csv('./Data/full_year/20-21_logs.csv')
else:
    ay20_21 = read_year('20-21')
    ay20_21.to_csv('./Data/full_year/20-21_logs.csv',index=False)
#%%
if os.path.isfile('./Data/full_year/21-22_logs.csv'):
    ay21_22 = pd.read_csv('./Data/full_year/21-22_logs.csv')
else:
    ay21_22 = read_year('21-22')
    ay21_22.to_csv('./Data/full_year/21-22_logs.csv',index=False)
#%%
if os.path.isfile('./Data/full_year/22-23_logs.csv'):
    ay22_23 = pd.read_csv('./Data/full_year/22-23_logs.csv')
else:
    ay22_23 = read_year('22-23')
    ay22_23.to_csv('./Data/full_year/22-23_logs.csv',index=False)
#%%
if os.path.isfile('./Data/full_year/23-24_logs.csv'):
    ay23_24 = pd.read_csv('./Data/full_year/23-24_logs.csv')
else:
    ay23_24 = read_year('23-24')
    ay23_24.to_csv('./Data/full_year/23-24_logs.csv',index=False)
#%%
all_years = pd.concat([ay19_20, ay20_21, ay21_22, ay22_23, ay23_24], ignore_index=True)
#%%
al_ids = pd.DataFrame(all_years.assignment_log_id.unique())
sampled_ids = al_ids.sample(n=100000).to_numpy().reshape(-1)
sample = all_years[all_years['assignment_log_id'].isin(sampled_ids)]
sample.to_csv('./Data/samples/validation_sample.csv', index=False)