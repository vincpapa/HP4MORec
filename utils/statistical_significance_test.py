from scipy.stats import ttest_ind, mannwhitneyu
import numpy as np


def mean_diff(x, y):
    return np.mean(x) - np.mean(y)


def statistical_significance(data1, data2):
    t_stat, p_mu_ttest = ttest_ind(data1, data2, equal_var=False)
    p_sigma = mannwhitneyu(data1, data2).pvalue

    return p_mu_ttest, p_sigma