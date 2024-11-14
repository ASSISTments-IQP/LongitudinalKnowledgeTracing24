#%%
import pandas as pd
from tqdm import tqdm
import os
#%% md
# # Split dataset into different dataframes for each year
#%%
def read_year(year):
    res = []

    with pd.read_csv("./Data/all_years_problem_logs.csv",chunksize=10**6) as read:
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
y_dict = {
    '19-20': ay19_20,
    '20-21': ay20_21,
    '21-22': ay21_22,
    '22-23': ay22_23,
    '23-24': ay23_24
}
#%%
for key, val in y_dict.items():
    high_interact_users = val.groupby(by=['user_xid']).filter(lambda x: len(x) > 5).user_xid.unique()
    high_interact_df = val[val['user_xid'].isin(high_interact_users)].copy()
    y_dict[key] = val
#%% md
# Summary Statistics for each academic year
#%%
for key, val in y_dict.items():
    print("Academic year ",key,':')
    print(len(val.assignment_log_id.unique()),' unique assignment log ids')
    print(len(val),' unique problem logs')
    print(len(val.user_xid.unique()),' unique users')
    print(len(val.skill_id.unique()),' unique skills')
    print(val.groupby(by=['user_xid']).size().mean(), ' average number of problems per user')
    print('Avg Correctness Value: ', val.discrete_score.mean())
#%%
for key, val in y_dict.items():
    assignment_log_ids = pd.DataFrame(val.assignment_log_id.unique())
    for i in tqdm(range(10)):
        sample_log_ids = assignment_log_ids.sample(n=50000).to_numpy().reshape(-1)
        sample = val[val['assignment_log_id'].isin(sample_log_ids)]
        sample.to_csv(f'./Data/samples/{key}/sample{i+1}.csv', index=False)
#%%
all_years = pd.concat(y_dict.values())
#%%
al_ids = pd.DataFrame(all_years.assignment_log_id.unique())
sampled_ids = al_ids.sample(n=100000).to_numpy().reshape(-1)
sample = all_years[all_years['assignment_log_id'].isin(sampled_ids)]
sample.to_csv('./Data/samples/validation_sample.csv', index=False)