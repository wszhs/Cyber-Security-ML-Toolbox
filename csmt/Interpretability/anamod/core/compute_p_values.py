"""Computes p-values for paired statistical tests over input vectors"""

import numpy as np
from numpy import asarray, compress, sqrt
from scipy.stats import find_repeats, rankdata, norm, ttest_rel

from anamod.core import constants, utils


def compute_empirical_p_value(baseline_loss, perturbed_loss, statistic):
    """Compute Monte Carlo estimate of empirical permutation-based p-value"""
    num_instances, num_permutations = perturbed_loss.shape
    if statistic == constants.MEAN_LOSS:
        baseline_statistic = np.mean(baseline_loss)
        perturbed_statistic = np.mean(perturbed_loss, axis=0)
    elif statistic == constants.MEAN_LOG_LOSS:
        baseline_statistic = np.mean(np.log(baseline_loss))
        perturbed_statistic = np.mean(np.log(perturbed_loss), axis=0)
    elif statistic == constants.MEDIAN_LOSS:
        baseline_statistic = np.median(baseline_loss)
        perturbed_statistic = np.median(perturbed_loss, axis=0)
    elif statistic == constants.RELATIVE_MEAN_LOSS:
        perturbed_statistic = np.zeros(num_permutations)
        for kidx in range(num_permutations):
            normalized_loss = np.divide(perturbed_loss[:, kidx], baseline_loss)
            perturbed_statistic[kidx] = np.mean(normalized_loss)
        baseline_statistic = 1
    elif statistic == constants.SIGN_LOSS:
        threshold = num_instances // 2
        perturbed_statistic = np.zeros(num_permutations)
        for kidx in range(num_permutations):
            count = sum(perturbed_loss[:, kidx] > baseline_loss + 1e-10)
            if count > threshold:
                perturbed_statistic[kidx] = 1
            elif count == threshold:
                perturbed_statistic[kidx] = 0
            else:
                perturbed_statistic[kidx] = -1
        baseline_statistic = 0
    else:
        raise ValueError(f"Unknown statistic {statistic}")
    # Baseline statistic should be smaller to reject null
    return (1 + sum(perturbed_statistic <= baseline_statistic + 1e-10)) / (1 + num_permutations)


def compute_p_value(baseline, perturbed, test=constants.PAIRED_TTEST, alternative=constants.TWOSIDED):
    """Compute p-value using paired difference test on input numpy arrays"""
    # TODO: Implement one-sided t-tests
    baseline = utils.round_value(baseline, decimals=15)
    perturbed = utils.round_value(perturbed, decimals=15)
    # Perform statistical test
    valid_tests = [constants.PAIRED_TTEST, constants.WILCOXON_TEST]
    assert test in valid_tests, f"Invalid test name {test}"
    if test == constants.PAIRED_TTEST:
        # Two-tailed paired t-test
        pvalue = ttest_rel(baseline, perturbed).pvalue
        if np.isnan(pvalue):
            # Identical vectors
            pvalue = 1.0
        return pvalue
    # One-tailed Wilcoxon signed-rank test
    return wilcoxon_test(baseline, perturbed, alternative=alternative)


def wilcoxon_test(x, y, alternative):
    """
    One-sided Wilcoxon signed-rank test derived from Scipy's two-sided test
    e.g. for alternative == constants.LESS, rejecting the null means that median difference x - y < 0
    Returns p-value
    """
    # TODO: add unit tests to verify results identical to R's Wilcoxon test for a host of input values
    # pylint: disable = invalid-name, too-many-locals
    x, y = map(asarray, (x, y))
    d = x - y

    d = compress(np.not_equal(d, 0), d, axis=-1)

    count = len(d)

    r = rankdata(abs(d))
    T = np.sum((d > 0) * r, axis=0)

    mn = count * (count + 1.) * 0.25
    se = count * (count + 1.) * (2. * count + 1.)

    if se < 1e-20:
        return 1.  # Degenerate case

    _, repnum = find_repeats(r)
    if repnum.size != 0:
        # Correction for repeated elements.
        se -= 0.5 * (repnum * (repnum * repnum - 1)).sum()

    se = sqrt(se / 24)
    if alternative == constants.LESS:
        correction = -0.5
    elif alternative == constants.GREATER:
        correction = 0.5
    else:
        correction = 0.5 * np.sign(T - mn)  # two-sided

    z = (T - mn - correction) / se

    if alternative == constants.LESS:
        return norm.cdf(z)
    if alternative == constants.GREATER:
        return norm.sf(z)
    return 2 * min(norm.cdf(z), norm.sf(z))  # two-sided


def bh_procedure(pvalues, significance_level):
    """Return adjusted p-values and rejected hypotheses computed according to Benjamini Hochberg procedure"""
    # pylint: disable = invalid-name
    m = len(pvalues)
    hypotheses = list(zip(range(m), pvalues))
    hypotheses.sort(key=lambda x: x[1])
    max_idx = 0
    adjusted_pvalues = np.ones(m)
    rejected_hypotheses = [False] * m
    for idx, hypothesis in enumerate(hypotheses):
        _, pvalue = hypothesis
        i = idx + 1
        adjusted_pvalues[idx] = m / i * pvalue
        critical_constant = i * significance_level / m
        if pvalue < critical_constant:
            max_idx = i
    for idx in range(max_idx):
        rejected_hypotheses[idx] = True
    for idx in reversed(range(m - 1)):
        # Adjusted pvalues - see http://www.biostathandbook.com/multiplecomparisons.html
        adjusted_pvalues[idx] = min(adjusted_pvalues[idx], adjusted_pvalues[idx + 1])
    data = sorted(zip(hypotheses, adjusted_pvalues, rejected_hypotheses), key=lambda elem: elem[0][0])
    _, adjusted_pvalues, rejected_hypotheses = zip(*data)
    return adjusted_pvalues, rejected_hypotheses
