"""Error models: Gates execution error, state dispersions, process noise."""

from __future__ import annotations

import numpy as np
import pytest

from gmat_uq import ManeuverErrorModel, NavigationErrorModel, ProcessNoise, StateDispersion
from gmat_uq.linalg import check_covariance, covariance_factor, perpendicular_basis, whitening_matrix
from gmat_uq.models import local_frame_rotation, rotate_covariance_to_inertial
from gmat_uq.stats import compare_covariance

GATES = ManeuverErrorModel(
    fixed_magnitude_sigma=2e-5,
    proportional_magnitude_sigma=0.01,
    fixed_pointing_sigma=1e-5,
    proportional_pointing_sigma=0.02,
)


# -- linear algebra helpers -----------------------------------------------------------------------
def test_covariance_factor_reproduces_positive_definite_matrix():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((6, 6))
    cov = a @ a.T
    factor = covariance_factor(cov)
    assert np.allclose(factor @ factor.T, cov, rtol=1e-12, atol=1e-12)
    assert np.allclose(factor, np.tril(factor))  # Cholesky for the definite case


def test_covariance_factor_handles_semidefinite_and_zero():
    vector = np.array([1.0, 2.0, 0.0, 0.0, -1.0, 0.5])
    cov = np.outer(vector, vector)  # rank one
    factor = covariance_factor(cov)
    assert np.allclose(factor @ factor.T, cov, atol=1e-12)
    assert not covariance_factor(np.zeros((6, 6))).any()


def test_check_covariance_rejects_bad_input():
    with pytest.raises(ValueError, match="6x6"):
        check_covariance(np.eye(3), 6)
    with pytest.raises(ValueError, match="symmetric"):
        bad = np.eye(6)
        bad[0, 1] = 1.0
        check_covariance(bad, 6)
    with pytest.raises(ValueError, match="positive semi-definite"):
        check_covariance(-np.eye(6), 6)
    with pytest.raises(ValueError, match="non-finite"):
        check_covariance(np.full((6, 6), np.nan), 6)


def test_perpendicular_basis_is_orthonormal_and_right_handed():
    rng = np.random.default_rng(2)
    directions = rng.standard_normal((200, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    directions = np.vstack([directions, np.eye(3), -np.eye(3)])
    first, second = perpendicular_basis(directions)
    assert np.allclose(np.einsum("ij,ij->i", first, directions), 0.0, atol=1e-12)
    assert np.allclose(np.einsum("ij,ij->i", second, directions), 0.0, atol=1e-12)
    assert np.allclose(np.einsum("ij,ij->i", first, second), 0.0, atol=1e-12)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(second, axis=1), 1.0)
    assert np.allclose(np.cross(directions, first), second, atol=1e-12)


def test_whitening_matrix_drops_null_directions():
    cov = np.diag([4.0, 1.0, 0.0])
    whitener = whitening_matrix(cov)
    assert whitener.shape == (2, 3)
    assert np.allclose(whitener @ cov @ whitener.T, np.eye(2))
    assert whitening_matrix(np.zeros((3, 3))).shape == (0, 3)


# -- Gates model ------------------------------------------------------------------------------------
def test_gates_covariance_has_the_documented_structure():
    dv = np.array([0.03, -0.04, 0.12])
    magnitude = np.linalg.norm(dv)
    unit = dv / magnitude
    cov = GATES.covariance(dv)
    along = np.sqrt(GATES.fixed_magnitude_sigma**2 + (GATES.proportional_magnitude_sigma * magnitude) ** 2)
    cross = np.sqrt(GATES.fixed_pointing_sigma**2 + (GATES.proportional_pointing_sigma * magnitude) ** 2)
    assert np.isclose(unit @ cov @ unit, along**2)
    first, second = perpendicular_basis(unit)
    assert np.isclose(first @ cov @ first, cross**2)
    assert np.isclose(second @ cov @ second, cross**2)
    assert np.isclose(np.trace(cov), along**2 + 2 * cross**2)


def test_gates_sampler_matches_its_analytic_covariance():
    """Validation test 1: 100,000 draws are consistent with the formula."""
    dv = np.array([0.08, 0.02, -0.05])
    rng = np.random.default_rng(12345)
    errors = GATES.sample(np.tile(dv, (100_000, 1)), rng)
    comparison = compare_covariance(errors, GATES.covariance(dv))
    assert comparison.p_value > 0.01  # consistent at the 99% level
    assert comparison.mean_p_value > 0.01
    assert comparison.max_sigma_deviation < 0.01


def test_gates_apply_is_vectorised_consistently():
    rng = np.random.default_rng(3)
    dv = rng.standard_normal((20, 3)) * 0.05
    normals = rng.standard_normal((20, 3))
    batch = GATES.apply(dv, normals)
    for row in range(20):
        assert np.array_equal(batch[row], GATES.apply(dv[row], normals[row]))


def test_gates_zero_delta_v_is_isotropic_and_keeps_total_variance():
    cov = GATES.covariance(np.zeros(3))
    expected = (GATES.fixed_magnitude_sigma**2 + 2 * GATES.fixed_pointing_sigma**2) / 3
    assert np.allclose(cov, expected * np.eye(3))
    normals = np.array([[1.0, -2.0, 0.5]])
    assert np.allclose(GATES.apply(np.zeros((1, 3)), normals), np.sqrt(expected) * normals)


def test_gates_expected_covariance_without_command_dispersion_is_the_plain_covariance():
    dv = np.array([0.01, 0.02, -0.03])
    assert np.allclose(GATES.expected_covariance(dv), GATES.covariance(dv))
    assert np.allclose(GATES.expected_covariance(dv, np.zeros((3, 3))), GATES.covariance(dv))


def test_gates_expected_covariance_proportional_terms_are_exact():
    model = ManeuverErrorModel(proportional_magnitude_sigma=0.02, proportional_pointing_sigma=0.01)
    mean = np.array([0.02, -0.01, 0.005])
    cov = np.diag([1e-4, 4e-4, 2.5e-5])
    second = cov + np.outer(mean, mean)
    expected = 0.02**2 * second + 0.01**2 * (np.trace(second) * np.eye(3) - second)
    assert np.allclose(model.expected_covariance(mean, cov), expected, rtol=1e-12, atol=0.0)


def test_gates_expected_covariance_agrees_with_brute_force():
    mean = np.array([0.004, 0.0, -0.002])
    command_cov = np.diag([2e-5, 1e-5, 3e-5])
    rng = np.random.default_rng(99)
    commands = mean + rng.standard_normal((400_000, 3)) @ np.linalg.cholesky(command_cov).T
    errors = GATES.sample(commands, rng)
    brute = errors.T @ errors / errors.shape[0]
    expected = GATES.expected_covariance(mean, command_cov)
    assert np.allclose(np.diag(brute), np.diag(expected), rtol=0.02)
    assert np.isclose(np.trace(brute), np.trace(expected), rtol=0.01)


def test_gates_rejects_negative_sigmas():
    with pytest.raises(ValueError):
        ManeuverErrorModel(fixed_magnitude_sigma=-1.0)
    with pytest.raises(ValueError):
        ManeuverErrorModel(proportional_pointing_sigma=float("nan"))
    assert ManeuverErrorModel().is_zero
    assert not GATES.is_zero


# -- state dispersions --------------------------------------------------------------------------------
def test_state_dispersion_from_sigmas_and_sampling():
    sigmas = [1.0, 2.0, 3.0, 1e-3, 2e-3, 3e-3]
    dispersion = StateDispersion.from_sigmas(sigmas)
    assert np.allclose(np.sqrt(np.diag(dispersion.covariance)), sigmas)
    assert np.allclose(StateDispersion.from_sigmas([2.0, 1e-3]).covariance, np.diag([4.0] * 3 + [1e-6] * 3))
    rng = np.random.default_rng(5)
    samples = dispersion.sample(rng, 50_000)
    assert compare_covariance(samples, dispersion.covariance).max_sigma_deviation < 0.02
    assert dispersion.sample(rng).shape == (6,)


def test_state_dispersion_is_immutable_and_validated():
    dispersion = StateDispersion(np.eye(6))
    with pytest.raises(ValueError):
        dispersion.covariance[0, 0] = 5.0
    with pytest.raises(ValueError):
        StateDispersion(np.eye(5))
    with pytest.raises(ValueError):
        StateDispersion.from_sigmas([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        StateDispersion.from_sigmas([-1.0] * 6)
    assert StateDispersion(np.eye(6)) == StateDispersion(np.eye(6))
    assert StateDispersion(np.eye(6)) != NavigationErrorModel(np.eye(6))


def test_local_frames_are_rotations_with_the_expected_axes():
    state = np.array([7000.0, 100.0, -50.0, 0.2, 7.4, 1.1])
    for frame in ("RIC", "VNB"):
        rotation = local_frame_rotation(state, frame)
        assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(rotation), 1.0)
    ric = local_frame_rotation(state, "RIC")
    assert np.allclose(ric[:, 0], state[:3] / np.linalg.norm(state[:3]))
    vnb = local_frame_rotation(state, "VNB")
    assert np.allclose(vnb[:, 0], state[3:] / np.linalg.norm(state[3:]))
    assert np.allclose(local_frame_rotation(state, "inertial"), np.eye(3))
    with pytest.raises(ValueError):
        local_frame_rotation(state, "XYZ")
    with pytest.raises(ValueError):
        local_frame_rotation(np.array([1.0, 0, 0, 2.0, 0, 0]), "RIC")  # rectilinear


def test_rotating_a_local_covariance_puts_the_radial_sigma_along_the_position():
    state = np.array([0.0, 8000.0, 0.0, -7.0, 0.0, 0.5])
    local = np.diag([9.0, 1.0, 0.25, 1e-6, 4e-6, 9e-6])  # radial sigma 3 km
    inertial = rotate_covariance_to_inertial(local, state, "RIC")
    radial = state[:3] / np.linalg.norm(state[:3])
    assert np.isclose(radial @ inertial[:3, :3] @ radial, 9.0)
    assert np.isclose(np.trace(inertial), np.trace(local))
    dispersion = StateDispersion.from_local(local, state, "RIC")
    assert np.allclose(dispersion.covariance, inertial)


def test_state_dispersion_from_file(tmp_path):
    cov = np.diag([1.0, 2.0, 3.0, 1e-6, 2e-6, 3e-6])
    path = tmp_path / "cov.txt"
    lines = ["% injection covariance", "# km, km/s"] + [", ".join(f"{v:.12e}" for v in row) for row in cov]
    path.write_text("\n".join(lines), encoding="utf-8")
    assert np.allclose(StateDispersion.from_file(path).covariance, cov)


# -- process noise ---------------------------------------------------------------------------------------
def test_process_noise_matches_the_snc_formula():
    noise = ProcessNoise((1e-6, 2e-6, 3e-6), max_step=300.0)
    dt = 120.0
    q = noise.covariance(dt)
    for axis, sigma in enumerate((1e-6, 2e-6, 3e-6)):
        assert np.isclose(q[axis, axis], sigma**2 * dt**3 / 3)
        assert np.isclose(q[axis, axis + 3], sigma**2 * dt**2 / 2)
        assert np.isclose(q[axis + 3, axis + 3], sigma**2 * dt)
    assert np.allclose(q, q.T)
    assert np.linalg.eigvalsh(q).min() > -1e-25
    assert np.allclose(noise.covariance(-dt), q)
    with pytest.raises(ValueError):
        ProcessNoise((1e-6, -1e-6, 0.0))
    with pytest.raises(ValueError):
        ProcessNoise((1e-6, 1e-6, 1e-6), max_step=0.0)


def test_covariance_files_in_the_layouts_gmat_writes(tmp_path):
    first = np.diag([1.0, 2.0, 3.0, 1e-6, 2e-6, 3e-6])
    second = 4.0 * first
    one_row = tmp_path / "one_row.txt"
    one_row.write_text(" ".join(repr(float(v)) for v in first.ravel()) + "\n", encoding="utf-8")
    assert np.array_equal(StateDispersion.from_file(one_row).covariance, first)
    blocks = tmp_path / "blocks.txt"
    lines = [" ".join(repr(float(v)) for v in row) for row in np.vstack([first, second])]
    blocks.write_text("\n".join(lines), encoding="utf-8")
    assert np.array_equal(StateDispersion.from_file(blocks).covariance, second)  # the last one by default
    assert np.array_equal(StateDispersion.from_file(blocks, block=0).covariance, first)
    with pytest.raises(ValueError, match="block 5 does not exist"):
        StateDispersion.from_file(blocks, block=5)
    short = tmp_path / "short.txt"
    short.write_text("1 2 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="holds 3 numbers"):
        StateDispersion.from_file(short)
    text = tmp_path / "text.txt"
    text.write_text("01 Jan 2026 1 2 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1: not a number"):
        StateDispersion.from_file(text)


def test_reads_a_covariance_written_by_gmat_batch_estimator():
    """``Report Rep EstSat.EarthMJ2000Eq.OrbitErrorCovariance`` after ``RunEstimator`` (GMAT R2026a sample)."""
    from pathlib import Path

    path = Path(__file__).parent / "data" / "gmat_batch_estimator_covariance.txt"
    knowledge = NavigationErrorModel.from_file(path)
    sigmas = np.sqrt(np.diag(knowledge.covariance))
    assert np.allclose(sigmas[:3], [1.7926e-4, 2.9299e-4, 5.0358e-4], rtol=1e-4)  # km
    assert np.all(np.linalg.eigvalsh(knowledge.covariance) > 0.0)


def test_gates_1963_model_is_reproduced():
    """Validation test 8: the report's construction, closed form and stated properties."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "validation"))
    try:
        import gates1963
    finally:
        sys.path.pop(0)
    result = gates1963.reproduce(samples=150_000, seed=7)
    assert result["passed"], result["checks"]
