"""Bootstrap confidence intervals (SPEC §6.5). Percentile method throughout.

Conventions:
- `None` values (undefined metrics, e.g. stability of an invalid build) are dropped; `n` reports how many were used.
- Rates (valid, stable) are means of 0/1 values. Mean/min stability, CLIP, DINO are passed in already
  restricted to valid builds (invalid builds carry None for those fields).
"""
from collections import defaultdict
from typing import NamedTuple

import numpy as np

from midway.config import BOOTSTRAP_N, BOOTSTRAP_SEED


class CI(NamedTuple):
    """Point estimate with a percentile CI. `mean` holds the median when a median was bootstrapped."""
    mean: float | None
    lo: float | None
    hi: float | None
    n: int


def _percentile_ci(point: float, boot_stats: np.ndarray, alpha: float, n: int) -> CI:
    lo, hi = np.percentile(boot_stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return CI(float(point), float(lo), float(hi), n)


def bootstrap_ci(values, n: int = BOOTSTRAP_N, alpha: float = 0.05, rng_seed: int = BOOTSTRAP_SEED) -> CI:
    """Mean of `values` with a (1 - alpha) percentile bootstrap CI.

    Resamples the non-None values with replacement `n` times and takes the alpha/2 and 1 - alpha/2
    percentiles of the resampled means.
    """
    x = np.array([v for v in values if v is not None], dtype=float)
    if len(x) == 0:
        return CI(None, None, None, 0)
    rng = np.random.default_rng(rng_seed)
    boot_means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return _percentile_ci(x.mean(), boot_means, alpha, len(x))


def cluster_bootstrap_ci(records: list[dict], key: str, statistic: str = 'mean', n: int = BOOTSTRAP_N,
                         alpha: float = 0.05, rng_seed: int = BOOTSTRAP_SEED) -> CI:
    """Mean (or median) of `record[key]` over records, with a bootstrap that resamples prompts, not rows.

    Use when a run set has several seeds per prompt: all of a prompt's seeds (records sharing an
    `object_id`) are kept together in each resample. The statistic is the pooled mean
    (sum of values in the resampled prompts / number of values in them). With one seed per prompt
    this is the same as `bootstrap_ci`.

    statistic='median': the median of all values in the resampled prompts (used for rollbacks and s/build).
    """
    by_prompt = defaultdict(list)
    for r in records:
        if r.get(key) is not None:
            by_prompt[r['object_id']].append(float(r[key]))
    if not by_prompt:
        return CI(None, None, None, 0)
    if statistic == 'median':
        return _cluster_median_ci(list(by_prompt.values()), n, alpha, rng_seed)
    sums = np.array([sum(v) for v in by_prompt.values()])
    counts = np.array([len(v) for v in by_prompt.values()])
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, len(sums), size=(n, len(sums)))
    boot_means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return _percentile_ci(sums.sum() / counts.sum(), boot_means, alpha, int(counts.sum()))


def _cluster_median_ci(groups: list[list[float]], n: int, alpha: float, rng_seed: int) -> CI:
    """Cluster bootstrap of the median: resample prompts, pool their values, take the median."""
    groups = [np.array(g) for g in groups]
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, len(groups), size=(n, len(groups)))
    boot_medians = np.array([np.median(np.concatenate([groups[i] for i in row])) for row in idx])
    all_values = np.concatenate(groups)
    return _percentile_ci(np.median(all_values), boot_medians, alpha, len(all_values))


def paired_diff_ci(a: dict, b: dict, n: int = BOOTSTRAP_N, alpha: float = 0.05,
                   rng_seed: int = BOOTSTRAP_SEED) -> CI:
    """Mean paired difference a - b with a bootstrap CI.

    `a` and `b` map a pairing key, e.g. (object_id, seed), to a value. Only keys present in both with
    non-None values are used; the bootstrap resamples those pairs.
    """
    shared = [k for k in a if k in b and a[k] is not None and b[k] is not None]
    return bootstrap_ci([a[k] - b[k] for k in shared], n=n, alpha=alpha, rng_seed=rng_seed)
