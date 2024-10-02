import pandas as pd
import os
if not os.path.exists('sakt/data'):
    os.makedirs('sakt/data')
df = pd.read_csv('../23-24-problem_logs.csv')
df1 = df.sample(n = 25000, random_state= 102)
df = df.sample(n = 50000, random_state= 69)
df.to_csv('sakt/data/subset_50000_train.csv', index=False)
df1.to_csv('sakt/data/subset_25000_test.csv')
print('saved')
