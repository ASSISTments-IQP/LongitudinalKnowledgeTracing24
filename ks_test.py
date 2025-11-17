import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests


def conv_to_arr(df: pd.DataFrame) -> np.ndarray:
    return df.values

def calculate_psi(base_df, comparison_df):
    pass

def calculate_KL_Divergence(base_df, comparison_df):
    pass

def calculate_JS_Divergence(base_df, comparison_df):
    pass


def test_distributional_similarity(base_df: pd.DataFrame, comparison_df: pd.DataFrame, col1, col2):
    assert base_df.shape[1] == comparison_df.shape[1]
    all_obs = pd.concat([base_df,comparison_df])

    obs = pd.crosstab(all_obs[col1],all_obs[col2],margins=False)

    res = chi2_contingency(obs.to_numpy().T)

    return res.pvalue, res.statistic

#fake test case

if __name__ == "__main__":
    years = ['19-20', '20-21', '21-22', '22-23', '23-24']
    sample_nums = range(1, 11)
    col2 = 'discrete_score'

    sample_dict = {}
    for year in years:
        y_dict = {}
        for n in sample_nums:
            y_dict[n] = pd.read_csv(f'./Data/samples/{year}/sample{str(n)}.csv')[[col2,'academic_year']]
        sample_dict[year] = y_dict

    # WY tests
    pvals = []
    n_wy = 0
    chi2_stats = []
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
            test['academic_year'] = 'a'

            cur_pval, cur_chi2 = test_distributional_similarity(train, test, 'sample_num', col2)
            pvals.append(cur_pval)
            chi2_stats.append(cur_chi2)
            ++n_wy

    # CY tests
    for train_y_idx in range(4):
        train_year = years[train_y_idx]
        for test_y_idx in range(train_y_idx+1,5):
            test_year = years[test_y_idx]
            for s in tqdm(sample_nums,desc=f'CY {train_year}/{test_year}'):
                cur_pval, cur_chi2 = test_distributional_similarity(sample_dict[train_year][s], sample_dict[test_year][s], 'academic_year', col2)
                pvals.append(cur_pval)
                chi2_stats.append(cur_chi2)

    pvals = np.array(pvals)
    print (pvals.shape)
    print(a.shape for a in chi2_stats)
    chi2_stats = np.array(chi2_stats)

    valid = np.isfinite(pvals)
    valid_p_vals = pvals[valid]
    valid_chi2_stats = chi2_stats[valid]
    num_sig_raw = np.sum(valid_p_vals < 0.05)
    valid_len = len(valid_p_vals)
    print(valid_len," Valid p values out of ",len(pvals))
    if valid_len > 0:
        rejected, p_corrected, _, _ = multipletests(valid_p_vals, alpha = 0.05, method = 'fdr_by')
        wy_rej = rejected[:n_wy]
        cy_rej = rejected[n_wy:]
        wy_r_sum = np.sum(wy_rej)
        cy_r_sum = np.sum(cy_rej)
        num_sig_corrected = np.sum(rejected)
    else:
        num_sig_corrected = 0
        p_corrected = np.array([])

    e_false_pos = 0.05 * valid_len
    print(f"feats tried: {valid.sum():d}")
    print(f"feats w/ p < {0.05:.3f} (raw): {num_sig_raw:d} "
          f"({num_sig_raw / valid.sum():.1%})")
    print(f"Expected false positives under null: {e_false_pos:.1f}")
    print(f"Features statistically significant after correction: {num_sig_corrected:d} "
          f"({num_sig_corrected / valid.sum():.1%})")
    print(f"wy tests which reject after correction: {wy_r_sum}")
    if len(valid_chi2_stats) > 0:
        print(f"Chi2 statistic - Mean: {np.mean(valid_chi2_stats):.3f}, "
              f"Median: {np.median(valid_chi2_stats):.3f}, "
              f"Max: {np.max(valid_chi2_stats):.3f}")

    if num_sig_corrected == 0:
        print("no statistically significant evidence that from diff distribution")
    elif num_sig_corrected <= e_false_pos:
        print("very few samples have p < 0.05, within false positive range")
    else:
        print("more than a few samples have p < 0.05. Not from same distribution.")