"""SPEC §9 item 9 (bootstrap part) and §6.5."""
import pytest

from midway.stats import bootstrap_ci, cluster_bootstrap_ci, paired_diff_ci


def test_constant_vector_ci_is_degenerate():
    ci = bootstrap_ci([0.7] * 50)
    assert ci.mean == ci.lo == ci.hi == pytest.approx(0.7)
    assert ci.n == 50


def test_bernoulli_024_n250_half_width_about_5_3_points():
    values = [1] * 60 + [0] * 190  # 24.0% of 250
    ci = bootstrap_ci(values)
    assert ci.mean == pytest.approx(0.24)
    half_width_points = 100 * (ci.hi - ci.lo) / 2
    assert half_width_points == pytest.approx(5.3, abs=1.0)


def test_nulls_are_ignored_and_n_reported():
    ci = bootstrap_ci([1.0, None, 0.0, None])
    assert ci.n == 2
    assert ci.mean == pytest.approx(0.5)


def test_all_null_gives_none():
    ci = bootstrap_ci([None, None])
    assert ci.n == 0 and ci.mean is None and ci.lo is None and ci.hi is None


def test_same_seed_is_reproducible():
    values = [0.1, 0.5, 0.9, 0.3, 0.7]
    assert bootstrap_ci(values) == bootstrap_ci(values)


def test_cluster_bootstrap_matches_plain_with_one_seed():
    records = [{'object_id': f'o{i}', 'v': float(i % 2)} for i in range(100)]
    plain = bootstrap_ci([r['v'] for r in records])
    cluster = cluster_bootstrap_ci(records, 'v')
    assert cluster.mean == pytest.approx(plain.mean)
    assert cluster.n == 100
    assert (cluster.hi - cluster.lo) == pytest.approx(plain.hi - plain.lo, abs=0.03)


def test_cluster_bootstrap_carries_seeds_together():
    # Each prompt's two seeds always agree, so the effective sample size is 50 prompts, not 100 rows.
    records = [{'object_id': f'o{i}', 'seed': s, 'v': float(i % 2)} for i in range(50) for s in (42, 43)]
    cluster = cluster_bootstrap_ci(records, 'v')
    naive = bootstrap_ci([r['v'] for r in records])
    assert (cluster.hi - cluster.lo) > (naive.hi - naive.lo)


def test_paired_diff_uses_only_shared_keys():
    a = {('o1', 42): 1.0, ('o2', 42): 1.0, ('o3', 42): 0.0}
    b = {('o1', 42): 0.0, ('o2', 42): 0.0, ('o4', 42): 1.0}
    ci = paired_diff_ci(a, b)
    assert ci.n == 2
    assert ci.mean == ci.lo == ci.hi == pytest.approx(1.0)


def test_cluster_median():
    records = [{'object_id': f'o{i}', 'v': float(i)} for i in range(11)]  # 0..10, median 5
    ci = cluster_bootstrap_ci(records, 'v', statistic='median', n=2000)
    assert ci.mean == 5.0 and ci.lo <= 5.0 <= ci.hi and ci.n == 11
