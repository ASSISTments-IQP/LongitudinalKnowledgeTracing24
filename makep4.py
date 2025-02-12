import pandas as pd
df = pd.read_csv("./Data/results_pt_3.csv")
def util(metric: str, df):
  df[f"bkt_{metric}"] = df.apply(lambda row: row[metric] if row["model"] == "BKT" else None, axis=1)
  df[f"pfa_{metric}"] = df.apply(lambda row: row[metric] if row["model"] == "PFA" else None, axis=1)
  df[f"dkt_{metric}"] = df.apply(lambda row: row[metric] if row["model"] == "DKT" else None, axis=1)
  df[f"sakt_kc_{metric}"] = df.apply(lambda row: row[metric] if row["model"] == "SAKT-KC" else None, axis=1)
  df[f"sakt_e_{metric}"] = df.apply(lambda row: row[metric] if row["model"] == "SAKT-E" else None, axis=1)
  df.drop([metric], axis = 1, inplace = True)

util('auc', df)
util('ll', df)
util('f1', df)


df_grouped = df.pivot_table(
    index=['train_year', 'eval_year', 'sample_num'],
    values=['bkt_auc',
            'pfa_auc',
            'dkt_auc',
            'sakt_kc_auc',
            'sakt_e_auc',
            'bkt_ll',
            'pfa_ll',
            'dkt_ll',
            'sakt_kc_ll',
            'sakt_e_ll',
            'bkt_f1', 'pfa_f1', 'dkt_f1', 'sakt_kc_f1', 'sakt_e_f1'],
    aggfunc='first'
).reset_index()

df_grouped.to_csv('results_pt_4.csv', index = False)
