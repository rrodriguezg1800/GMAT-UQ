"""Statistics: quantile intervals, Wilks sample size, covariance comparison, ellipses."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as scipy_stats

from gmat_uq.stats import (
    compare_covariance,
    error_ellipse,
    quantile_with_interval,
    sigma_ratio_interval,
    summarize_magnitudes,
    wilks_sample_size,
)


def test_wilks_sample_size_known_values():
    assert wilks_sample_size(0.99, 0.95) == 299
    assert wilks_sample_size(0.95, 0.95) == 59
    assert wilks_sample_size(0.99, 0.99) == 459
    assert wilks_sample_size(0.5, 0.5) == 1
    with pytest.raises(ValueError):
        wilks_sample_size(1.0, 0.95)
    with pytest.raises(ValueError):
        wilks_sample_size(0.99, 0.0)


def test_wilks_criterion_is_the_point_where_the_upper_bound_becomes_finite():
    rng = np.random.default_rng(0)
    below = quantile_with_interval(rng.standard_normal(100), 0.99, 0.95)
    assert np.isinf(below.upper)
    enough = quantile_with_interval(rng.standard_normal(2000), 0.99, 0.95)
    assert np.isfinite(enough.upper) and np.isfinite(enough.lower)
    assert enough.lower <= enough.value <= enough.upper
    assert enough.contains(enough.value) and not enough.contains(enough.upper + 1.0)
    assert enough.to_dict()["n_samples"] == 2000


def test_quantile_interval_has_the_advertised_coverage():
    rng = np.random.default_rng(7)
    true_value = scipy_stats.expon.ppf(0.9)
    trials = 2000
    covered = 0
    for _ in range(trials):
        estimate = quantile_with_interval(rng.exponential(size=400), 0.9, 0.95)
        covered += estimate.contains(true_value)
    coverage = covered / trials
    assert coverage >= 0.94  # nominal 0.95, minus Monte Carlo noise
    assert coverage <= 0.995  # and not uselessly wide


def test_quantile_estimate_validation():
    with pytest.raises(ValueError):
        quantile_with_interval([], 0.9)
    with pytest.raises(ValueError):
        quantile_with_interval([1.0, 2.0], 1.5)
    with pytest.raises(ValueError):
        quantile_with_interval([1.0, 2.0], 0.5, 1.0)


def test_compare_covariance_accepts_matching_samples():
    rng = np.random.default_rng(11)
    a = rng.standard_normal((6, 6))
    cov = a @ a.T
    samples = rng.multivariate_normal(np.zeros(6), cov, size=5000)
    result = compare_covariance(samples, cov)
    assert result.dimension == 6 and result.n_samples == 5000
    assert result.p_value > 0.001
    assert result.mean_p_value > 0.001
    assert result.max_sigma_deviation < 0.06
    assert result.degrees_of_freedom == 21
    assert "max_sigma_deviation" in result.to_dict()


def test_compare_covariance_p_values_are_uniform_under_the_null():
    rng = np.random.default_rng(13)
    cov = np.diag([4.0, 1.0, 0.25])
    p_values = [compare_covariance(rng.multivariate_normal(np.zeros(3), cov, size=300), cov).p_value for _ in range(400)]
    # A valid test rejects at the 5% level about 5% of the time.
    rejection_rate = np.mean(np.array(p_values) < 0.05)
    assert 0.02 < rejection_rate < 0.09


def test_compare_covariance_detects_a_mismatch():
    rng = np.random.default_rng(17)
    cov = np.eye(4)
    samples = rng.multivariate_normal(np.zeros(4), cov, size=4000)
    inflated = compare_covariance(samples * 1.2, cov)
    assert inflated.p_value < 1e-6
    assert 1.12 < inflated.sigma_ratio_max < 1.28
    shifted = compare_covariance(samples + 0.2, cov)
    assert shifted.mean_p_value < 1e-6
    assert np.isclose(shifted.mean_offset_sigma, 0.4, atol=0.06)


def test_compare_covariance_works_in_the_range_of_a_singular_reference():
    rng = np.random.default_rng(19)
    direction = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 2.0]])
    cov = direction.T @ direction  # rank two in three dimensions
    samples = rng.standard_normal((3000, 2)) @ direction
    result = compare_covariance(samples, cov)
    assert result.dimension == 2
    assert result.max_sigma_deviation < 0.06


def test_compare_covariance_validation():
    with pytest.raises(ValueError, match="two-dimensional"):
        compare_covariance(np.zeros(5), np.eye(1))
    with pytest.raises(ValueError, match="does not match"):
        compare_covariance(np.zeros((10, 3)), np.eye(2))
    with pytest.raises(ValueError, match="is zero"):
        compare_covariance(np.zeros((10, 2)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="need more than"):
        compare_covariance(np.zeros((3, 2)), np.eye(2))


def test_sigma_ratio_interval_brackets_the_ratio():
    ratio, lower, upper = sigma_ratio_interval(1.05, 1.0, 500)
    assert lower < ratio < upper
    assert np.isclose(ratio, 1.05)
    assert 0.98 < lower < 1.0 and 1.1 < upper < 1.13
    with pytest.raises(ValueError):
        sigma_ratio_interval(1.0, 0.0, 100)
    with pytest.raises(ValueError):
        sigma_ratio_interval(1.0, 1.0, 1)


def test_error_ellipse_axes_orientation_and_probability():
    ellipse = error_ellipse(np.diag([9.0, 1.0]), 1.0)
    assert np.isclose(ellipse.semi_major, 3.0) and np.isclose(ellipse.semi_minor, 1.0)
    assert np.isclose(ellipse.angle_deg, 0.0)
    assert np.isclose(ellipse.probability, 1 - np.exp(-0.5))
    rotated = error_ellipse(np.array([[5.0, 4.0], [4.0, 5.0]]), 3.0)
    assert np.isclose(rotated.semi_major, 9.0) and np.isclose(rotated.semi_minor, 3.0)
    assert np.isclose(rotated.angle_deg, 45.0)
    assert np.isclose(rotated.probability, 1 - np.exp(-4.5))
    assert np.isclose(error_ellipse(np.array([[5.0, -4.0], [-4.0, 5.0]])).angle_deg, -45.0)
    with pytest.raises(ValueError):
        error_ellipse(np.eye(3))


def test_ellipse_probability_matches_sampling():
    rng = np.random.default_rng(23)
    cov = np.array([[2.0, 0.6], [0.6, 0.5]])
    samples = rng.multivariate_normal(np.zeros(2), cov, size=200_000)
    inside = np.einsum("ij,jk,ik->i", samples, np.linalg.inv(cov), samples) <= 3.0**2
    assert np.isclose(inside.mean(), error_ellipse(cov, 3.0).probability, atol=2e-3)


def test_magnitude_summary_of_an_isotropic_correction_is_maxwell_distributed():
    # The magnitude of an isotropic 3-D Gaussian vector follows a Maxwell distribution.
    sigma = 0.002
    rng = np.random.default_rng(29)
    magnitudes = np.linalg.norm(rng.standard_normal((200_000, 3)) * sigma, axis=1)
    summary = summarize_magnitudes(magnitudes, 0.99, 0.95)
    assert np.isclose(summary["mean"], 2 * sigma * np.sqrt(2 / np.pi), rtol=5e-3)
    assert np.isclose(summary["quantile"]["value"], sigma * np.sqrt(scipy_stats.chi2.ppf(0.99, 3)), rtol=1e-2)
    assert summary["n_samples"] == 200_000 and summary["max"] >= summary["quantile"]["value"]
