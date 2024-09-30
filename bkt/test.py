from BKT_Model import BKTModel
import pandas as pd

bkt = BKTModel()
bkt.fit(pd.read_csv("samples.csv"))
