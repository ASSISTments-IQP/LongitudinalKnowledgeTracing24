from BKT_Model import BKTModel
import pandas as pd

df = pd.read_csv("Data/sample.csv")
bkt = BKTModel(n_iter=100)
bkt.fit(df)