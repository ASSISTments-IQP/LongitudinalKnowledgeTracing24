import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import ks_2samp
from statsmodels.stats.multitest import multipletests


def conv_to_arr(df: pd.DataFrame) -> np.ndarray:
    return df.values


def test_distributional_similarity(base_df: pd.DataFrame, comparison_df: pd.DataFrame, plot: bool = False, fname: str = "ks_test.png", alpha: float = 0.05):
    assert base_df.shape[1] == comparison_df.shape[1]
    base_arr, comp_arr = conv_to_arr(base_df), conv_to_arr(comparison_df)
    n_feats = base_arr.shape[1]
    ks_stats = np.empty(n_feats)
    p_vals = np.empty(n_feats)
    for j in tqdm(range(n_feats), desc="KS per feat" ):
        a = base_arr[:, j]
        b = comp_arr[:, j]
        if len(a) == 0 or len(b) == 0:
            ks_stats[j] = np.nan
            p_vals[j] = np.nan
            continue

        ks, p = ks_2samp(a, b)
        ks_stats[j] = ks
        p_vals[j] = p

    valid = np.isfinite(p_vals)
    valid_p_vals = p_vals[valid]
    valid_ks_stats = ks_stats[valid]
    num_sig_raw = np.sum(valid_p_vals < alpha)
    e_false_pos = alpha * len(valid_p_vals)

    if num_sig_raw > e_false_pos:
        plot = True

    if plot:
        if not fname:
            fname = "ks_test_stuff.png"
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.hist(valid_p_vals, bins = 50, density = True, label = 'Observed P Values')
        ax1.axhline(1.0, color='red', linestyle='--', alpha=0.7, label='Uniform (null)')
        ax1.axvline(alpha, ls="--", lw=1.5, color='black', label=f"p = {alpha}")
        ax1.set_xlabel("KS p-value")
        ax1.set_ylabel("Density")
        ax1.set_title("P-value Distribution")
        ax1.legend()
        ax2.hist(valid_ks_stats, bins=50, density=True, alpha=0.7)
        ax2.set_xlabel("KS statistic")
        ax2.set_ylabel("Density")
        ax2.set_title("KS Statistic Distribution")
        plt.tight_layout()
        plt.savefig(fname)

    return p_vals, ks_stats

#fake test case

if __name__ == "__main__":
    years = ['19-20', '20-21', '21-22', '22-23', '23-24']
    sample_nums = range(1, 11)

    sample_dict = {}
    for year in years:
        y_dict = {}
        for n in sample_nums:
            y_dict[n] = pd.read_csv(f'./Data/samples/{year}/sample{str(n)}.csv')[['skill_id','user_xid','old_problem_id','discrete_score']]
        sample_dict[year] = y_dict

    # WY tests
    pvals = []
    ks_stats = []
    for y in years:
        for train_samp in sample_nums:
            train = []
            test = []
            for s in sample_nums:
                if s == train_samp:
                    train = sample_dict[y][s]
                else:
                    test.append(sample_dict[y][s])

            test = pd.concat(test)

            cur_pval, cur_ks = test_distributional_similarity(train, test, fname=f'{y}-{str(train_samp)}')
            pvals.append(cur_pval)
            ks_stats.append(ks_stats)

    # CY tests
    for train_y_idx in range(4):
        train_year = years[train_y_idx]
        for test_y_idx in range(train_y_idx,5):
            test_year = years[test_y_idx]
            for s in sample_nums:
                cur_pval, cur_ks = test_distributional_similarity(sample_dict[train_year][s],sample_dict[test_year][s], fname=f'{train_year}-{test_year}-{str(s)}')
                pvals.append(cur_pval)
                ks_stats.append(cur_ks)

    valid = np.isfinite(pvals)
    valid_p_vals = pvals[valid]
    valid_ks_stats = ks_stats[valid]
    num_sig_raw = np.sum(valid_p_vals < 0.05)
    if len(valid_p_vals) > 0:
        rejected, p_corrected, _, _ = multipletests(valid_p_vals, alpha = 0.05, method = 'fdr_by')
        num_sig_corrected = np.sum(rejected)
    else:
        num_sig_corrected = 0
        p_corrected = np.array([])

    e_false_pos = 0.05 * len(valid_p_vals)
    print(f"feats tried: {valid.sum():d}")
    print(f"feats w/ p < {0.05:.3f} (raw): {num_sig_raw:d} "
          f"({num_sig_raw / valid.sum():.1%})")
    print(f"Expected false positives under null: {e_false_pos:.1f}")
    print(f"Features statistically significant after correction: {num_sig_corrected:d} "
          f"({num_sig_corrected / valid.sum():.1%})")
    if len(valid_ks_stats) > 0:
        print(f"KS statistic - Mean: {np.mean(valid_ks_stats):.3f}, "
              f"Median: {np.median(valid_ks_stats):.3f}, "
              f"Max: {np.max(valid_ks_stats):.3f}")

    if num_sig_corrected == 0:
        print("no statistically significant evidence that from diff distribution")
    elif num_sig_corrected <= e_false_pos:
        print("very few samples have p < 0.05, within false positive range")
    else:
        print("more than a few samples have p < 0.05. Not from same distribution.")