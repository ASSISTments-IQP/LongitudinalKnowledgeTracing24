import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests


def conv_to_arr(df: pd.DataFrame) -> np.ndarray:
    return df.values

def psi(pcts_1, pcts_2):
    return np.sum((pcts_1-pcts_2) * np.log(pcts_1 / pcts_2))

def KL_div(pcts_1, pcts_2):
    return np.sum(pcts_1 * np.log(pcts_1 / pcts_2))

def JS_div(pcts_1, pcts_2):
    mid_pcts = (pcts_1 + pcts_2) / 2
    return 0.5 * (KL_div(pcts_1, mid_pcts) + KL_div(pcts_2, mid_pcts))


def chi_square_test(base_df: pd.DataFrame, comparison_df: pd.DataFrame, col1, col2):
    assert base_df.shape[1] == comparison_df.shape[1]
    all_obs = pd.concat([base_df,comparison_df])

    obs = pd.crosstab(all_obs[col1],all_obs[col2],margins=False)

    res = chi2_contingency(obs.to_numpy().T)

    return res.pvalue, res.statistic

def handle_non_intersecting(df1, df2, column_name):
    df1_vals = df1[column_name].unique()
    df2_vals = df2[column_name].unique()
    in_both = np.intersect1d(df1_vals, df2_vals)

    df1_only = df1[~df1[column_name].isin(in_both)]
    df2_only = df2[~df2[column_name].isin(in_both)]

    if df1_only.shape[0] == 0 and df2_only.shape[0] == 0:
        return df1, df2

    if df1_only.shape[0] == 0:
        df1 = pd.concat([df1, pd.DataFrame([[-1, df1.iloc[0].academic_year, df1.iloc[0].sample_num]], columns=[column_name,'academic_year','sample_num'])])
    else:
        df1.loc[df1_only.index,column_name] = -1

    if df2_only.shape[0] == 0:
        df2 = pd.concat([df2, pd.DataFrame([[-1, df2.iloc[0].academic_year, df2.iloc[0].sample_num]], columns=[column_name,'academic_year','sample_num'])])
    else:
        df2.loc[df2_only.index,column_name] = -1

    return df1, df2

#fake test case

if __name__ == "__main__":
    years = ['19-20', '20-21', '21-22', '22-23', '23-24']
    sample_nums = range(1, 11)
    col2 = 'skill_id'

    sample_dict = {}
    for year in years:
        y_dict = {}
        for n in sample_nums:
            y_dict[n] = pd.read_csv(f'./Data/samples/{year}/sample{str(n)}.csv')[[col2,'academic_year']]
        sample_dict[year] = y_dict

    # WY tests
    res = []
    for y in years:
        for train_samp in tqdm(sample_nums, desc=f'WY {y}'):
            train = []
            test = []
            for s in sample_nums:
                sample_dict[y][s]['sample_num'] = s
                if s == train_samp:
                    train = sample_dict[y][s]
                else:
                    test.append(sample_dict[y][s])

            test = pd.concat(test)
            test.sample_num = 0
            test.academic_year = 'a'

            p, chi2 = chi_square_test(train, test, 'academic_year', col2)

            train, test = handle_non_intersecting(train, test, col2)

            train_pcts = np.array((train.groupby([col2]).value_counts() / train.shape[0]).tolist())
            test_pcts = np.array((test.groupby([col2]).value_counts() / test.shape[0]).tolist())

            res.append([y+'/'+str(train_samp),chi2, p,psi(train_pcts, test_pcts),KL_div(train_pcts, test_pcts),JS_div(train_pcts, test_pcts),True])

    # CY tests
    for train_y_idx in range(4):
        train_year = years[train_y_idx]
        for test_y_idx in range(train_y_idx+1,5):
            test_year = years[test_y_idx]
            for s in tqdm(sample_nums,desc=f'CY {train_year}/{test_year}'):
                train = sample_dict[train_year][s]
                test = sample_dict[test_year][s]
                p, chi2 = chi_square_test(train, test, 'academic_year', col2)

                train, test = handle_non_intersecting(train, test, col2)

                train_pcts = np.array((train.groupby([col2]).value_counts() / train.shape[0]).tolist())
                test_pcts = np.array((test.groupby([col2]).value_counts() / test.shape[0]).tolist())

                res.append([train_year+'/'+test_year+'/'+str(s), chi2, p, psi(train_pcts, test_pcts), KL_div(train_pcts, test_pcts),
                            JS_div(train_pcts, test_pcts), False])

    res_df = pd.DataFrame(res, columns=['year-sample', 'chi-squared','pvalue', 'psi', 'KL divergence', 'JS divergence', 'within-year?'])

    res_df.to_csv(f'./drift_statistics_{col2}.csv',index=False)



    # pvals = np.array(pvals)
    # print (pvals.shape)
    # print(a.shape for a in chi2_stats)
    # chi2_stats = np.array(chi2_stats)
    #
    # valid = np.isfinite(pvals)
    # valid_p_vals = pvals[valid]
    # valid_chi2_stats = chi2_stats[valid]
    # num_sig_raw = np.sum(valid_p_vals < 0.05)
    # valid_len = len(valid_p_vals)
    # print(valid_len," Valid p values out of ",len(pvals))
    # if valid_len > 0:
    #     rejected, p_corrected, _, _ = multipletests(valid_p_vals, alpha = 0.05, method = 'fdr_by')
    #     wy_rej = rejected[:n_wy]
    #     cy_rej = rejected[n_wy:]
    #     wy_r_sum = np.sum(wy_rej)
    #     cy_r_sum = np.sum(cy_rej)
    #     num_sig_corrected = np.sum(rejected)
    # else:
    #     num_sig_corrected = 0
    #     p_corrected = np.array([])
    #
    # e_false_pos = 0.05 * valid_len
    # print(f"feats tried: {valid.sum():d}")
    # print(f"feats w/ p < {0.05:.3f} (raw): {num_sig_raw:d} "
    #       f"({num_sig_raw / valid.sum():.1%})")
    # print(f"Expected false positives under null: {e_false_pos:.1f}")
    # print(f"Features statistically significant after correction: {num_sig_corrected:d} "
    #       f"({num_sig_corrected / valid.sum():.1%})")
    # print(f"wy tests which reject after correction: {wy_r_sum}")
    # if len(valid_chi2_stats) > 0:
    #     print(f"Chi2 statistic - Mean: {np.mean(valid_chi2_stats):.3f}, "
    #           f"Median: {np.median(valid_chi2_stats):.3f}, "
    #           f"Max: {np.max(valid_chi2_stats):.3f}")
    #
    # if num_sig_corrected == 0:
    #     print("no statistically significant evidence that from diff distribution")
    # elif num_sig_corrected <= e_false_pos:
    #     print("very few samples have p < 0.05, within false positive range")
    # else:
    #     print("more than a few samples have p < 0.05. Not from same distribution.")