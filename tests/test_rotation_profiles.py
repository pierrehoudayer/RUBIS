import numpy as np

from rubis.rotation_profiles import (
    lorentzian, 
    plateau, 
    solid,
    tabulated,
)


def test_solid_rotation_profile_is_constant():
    r = np.linspace(0.0, 1.0, 11)
    cos_theta = 0.37
    omega = 0.63

    profile = solid(
        r,
        cos_theta,
        omega,
        return_profile=True,
    )

    np.testing.assert_allclose(
        profile,
        omega,
        rtol=0.0,
        atol=0.0,
    )


def test_solid_rotation_potential():
    r = np.linspace(0.0, 1.0, 11)
    cos_theta = 0.37
    omega = 0.63

    potential, derivative = solid(r, cos_theta, omega)

    sin_theta_squared = 1.0 - cos_theta**2

    expected_potential = (
        -0.5 * omega**2 * r**2 * sin_theta_squared
    )
    expected_derivative = (
        -omega**2 * r * sin_theta_squared
    )

    np.testing.assert_allclose(
        potential,
        expected_potential,
        rtol=1.0e-14,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        derivative,
        expected_derivative,
        rtol=1.0e-14,
        atol=1.0e-15,
    )


def test_solid_rotation_derivative_matches_finite_difference():
    r = np.array([0.1, 0.4, 0.9])
    cos_theta = 0.37
    omega = 0.63
    step = 1.0e-6

    potential_plus, _ = solid(
        r + step,
        cos_theta,
        omega,
    )
    potential_minus, _ = solid(
        r - step,
        cos_theta,
        omega,
    )
    _, derivative = solid(r, cos_theta, omega)

    numerical_derivative = (
        potential_plus - potential_minus
    ) / (2.0 * step)

    np.testing.assert_allclose(
        derivative,
        numerical_derivative,
        rtol=1.0e-9,
        atol=1.0e-11,
    )
    
    
def test_lorentzian_rotation_profile():
    r = np.linspace(0.0, 1.0, 11)
    cos_theta = 0.37
    omega = 0.63
    alpha = 0.4

    profile = lorentzian(
        r,
        cos_theta,
        omega,
        alpha,
        return_profile=True,
    )

    cylindrical_radius_squared = (
        r**2 * (1.0 - cos_theta**2)
    )
    expected_profile = (
        omega
        * (1.0 + alpha)
        / (1.0 + alpha * cylindrical_radius_squared)
    )

    np.testing.assert_allclose(
        profile,
        expected_profile,
        rtol=1.0e-14,
        atol=1.0e-15,
    )


def test_lorentzian_reduces_to_solid_rotation():
    r = np.linspace(0.0, 1.0, 11)
    cos_theta = 0.37
    omega = 0.63

    lorentzian_potential, lorentzian_derivative = (
        lorentzian(r, cos_theta, omega, alpha=0.0)
    )
    solid_potential, solid_derivative = solid(
        r,
        cos_theta,
        omega,
    )

    np.testing.assert_allclose(
        lorentzian_potential,
        solid_potential,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        lorentzian_derivative,
        solid_derivative,
        rtol=0.0,
        atol=0.0,
    )


def test_lorentzian_derivative_matches_finite_difference():
    r = np.array([0.1, 0.4, 0.9])
    cos_theta = 0.37
    omega = 0.63
    alpha = 0.4
    step = 1.0e-6

    potential_plus, _ = lorentzian(
        r + step,
        cos_theta,
        omega,
        alpha,
    )
    potential_minus, _ = lorentzian(
        r - step,
        cos_theta,
        omega,
        alpha,
    )
    _, derivative = lorentzian(
        r,
        cos_theta,
        omega,
        alpha,
    )

    numerical_derivative = (
        potential_plus - potential_minus
    ) / (2.0 * step)

    np.testing.assert_allclose(
        derivative,
        numerical_derivative,
        rtol=1.0e-9,
        atol=1.0e-11,
    )
    
    
def test_plateau_rotation_profile_normalization():
    omega = 0.63
    alpha = 0.4
    scale = 0.25

    central_profile = plateau(
        np.array([0.0]),
        0.0,
        omega,
        alpha,
        scale,
        return_profile=True,
    )
    equatorial_profile = plateau(
        np.array([1.0]),
        0.0,
        omega,
        alpha,
        scale,
        return_profile=True,
    )

    np.testing.assert_allclose(
        central_profile,
        (1.0 + alpha) * omega,
        rtol=1.0e-14,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        equatorial_profile,
        omega,
        rtol=1.0e-14,
        atol=1.0e-15,
    )


def test_plateau_reduces_to_solid_rotation():
    r = np.linspace(0.0, 1.0, 11)
    cos_theta = 0.37
    omega = 0.63
    scale = 0.25

    plateau_profile = plateau(
        r,
        cos_theta,
        omega,
        alpha=0.0,
        scale=scale,
        return_profile=True,
    )
    solid_profile = solid(
        r,
        cos_theta,
        omega,
        return_profile=True,
    )

    plateau_potential, plateau_derivative = plateau(
        r,
        cos_theta,
        omega,
        alpha=0.0,
        scale=scale,
    )
    solid_potential, solid_derivative = solid(
        r,
        cos_theta,
        omega,
    )

    np.testing.assert_allclose(
        plateau_profile,
        solid_profile,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        plateau_potential,
        solid_potential,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        plateau_derivative,
        solid_derivative,
        rtol=0.0,
        atol=0.0,
    )


def test_plateau_potential_derivative_matches_finite_difference():
    r = np.array([0.1, 0.4, 0.9])
    cos_theta = 0.37
    omega = 0.63
    alpha = 0.4
    scale = 0.25
    step = 1.0e-6

    potential_plus, _ = plateau(
        r + step,
        cos_theta,
        omega,
        alpha,
        scale,
    )
    potential_minus, _ = plateau(
        r - step,
        cos_theta,
        omega,
        alpha,
        scale,
    )
    _, derivative = plateau(
        r,
        cos_theta,
        omega,
        alpha,
        scale,
    )

    numerical_derivative = (
        potential_plus - potential_minus
    ) / (2.0 * step)

    np.testing.assert_allclose(
        derivative,
        numerical_derivative,
        rtol=1.0e-8,
        atol=1.0e-10,
    )
    
    
def test_tabulated_rotation_profile(tmp_path):
    s = np.linspace(0.0, 1.0, 11)
    omega_data = 1.2 - 0.2*s**2

    path = tmp_path / "rotation.txt"
    np.savetxt(
        path,
        np.column_stack((
            s,
            omega_data,
        )),
    )

    omega_eq = 0.63
    rotation_law = tabulated(path)

    omega, domega_ds = rotation_law(
        s,
        0.0,
        omega_eq,
        return_profile=True,
        return_dprofile=True,
    )

    norm = omega_eq / omega_data[-1]

    np.testing.assert_allclose(
        omega,
        omega_data * norm,
        rtol=1.0e-13,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        domega_ds,
        -0.4*s * norm,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        omega[-1],
        omega_eq,
        rtol=1.0e-14,
        atol=1.0e-15,
    )


def test_tabulated_potential_derivative_matches_finite_difference(
    tmp_path,
):
    s = np.linspace(0.0, 1.0, 21)
    omega_data = 1.2 - 0.2*s**2

    path = tmp_path / "rotation.txt"
    np.savetxt(
        path,
        np.column_stack((
            s,
            omega_data,
        )),
    )

    rotation_law = tabulated(path)

    r = np.array([0.1, 0.4, 0.8])
    t = 0.37
    omega_eq = 0.63
    step = 1.0e-6

    potential_plus, _ = rotation_law(
        r + step,
        t,
        omega_eq,
    )
    potential_minus, _ = rotation_law(
        r - step,
        t,
        omega_eq,
    )
    _, derivative = rotation_law(
        r,
        t,
        omega_eq,
    )

    numerical_derivative = (
        potential_plus - potential_minus
    ) / (2*step)

    np.testing.assert_allclose(
        derivative,
        numerical_derivative,
        rtol=1.0e-9,
        atol=1.0e-10,
    )