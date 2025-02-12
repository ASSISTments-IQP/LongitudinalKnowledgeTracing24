import pandas as pd
df = pd.read_csv("results_pt_3.csv")

df["bkt_auc"] = df.apply(lambda row: row["auc"] if row["model"] == "BKT" else None, axis=1)
df["pfa_auc"] = df.apply(lambda row: row["auc"] if row["model"] == "PFA" else None, axis=1)
df["dkt_auc"] = df.apply(lambda row: row["auc"] if row["model"] == "DKT" else None, axis=1)
df["sakt_kc_auc"] = df.apply(lambda row: row["auc"] if row["model"] == "SAKT-KC" else None, axis=1)
df["sakt_e_auc"] = df.apply(lambda row: row["auc"] if row["model"] == "SAKT-E" else None, axis=1)

df = df.drop(['auc'], axis = 1)

df_grouped = df.pivot_table(
    index=['train_year', 'eval_year', 'sample_num'],
    values=['bkt_auc', 'pfa_auc', 'dkt_auc', 'sakt_kc_auc', 'sakt_e_auc', 'll', 'f1'],
    aggfunc='first'
).reset_index()
df_grouped.to_csv('results_pt_4.csv', index = False)
