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
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
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
    if len(valid_p_vals) > 0:
        rejected, p_corrected, _, _ = multipletests(valid_p_vals, alpha = alpha, method = 'holm')
        num_sig_corrected = np.sum(rejected)
    else:
        num_sig_corrected = 0
        p_corrected = np.array([])

    e_false_pos = alpha * len(valid_p_vals)
    print(f"feats tried: {valid.sum():d}")
    print(f"feats w/ p < {alpha:.3f} (raw): {num_sig_raw:d} "
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
        plt.show() #doesnt work on my machine since pyqt5 sucks. savefig is fine

        return {
        'ks_stats': ks_stats,
        'p_vals': p_vals,
        'p_corrected': p_corrected,
        'num_significant_raw': num_sig_raw,
        'num_significant_corrected': num_sig_corrected,
        'valid_mask': valid
    }

#fake test case

if __name__ == "__main__":
    df_1 = pd.read_csv(f'../Data/samples/{2023}/sample{3}.csv')
    df_2 = pd.read_csv(f'../Data/samples/{2024}/sample{7}.csv')
    ks_test_res = test_distributional_similarity(df_1, df_2, plot = True)
