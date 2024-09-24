import pandas as pd
import os
if not os.path.exists('sakt/data'):
    os.makedirs('sakt/data')
df = pd.read_csv('../23-24-problem_logs.csv')
df = df.sample(n = 750000, random_state= 69)
df.to_csv('sakt/data/subset.csv')