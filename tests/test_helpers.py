import numpy as np

from helpers import (
    find_domains,
    init_phi_c,
    valid_reciprocal_domain,
)
from rubis.rotation_profiles import lorentzian, plateau, solid


def test_find_domains_for_continuous_coordinate():
    zeta = np.linspace(0.0, 1.0, 5)

    domains = find_domains(zeta)

    assert domains.Nd == 1

    np.testing.assert_array_equal(
        domains.bounds,
        np.array([]),
    )
    np.testing.assert_array_equal(
        domains.edges,
        np.array([0, 5]),
    )
    np.testing.assert_array_equal(
        domains.sizes,
        np.array([5]),
    )
    np.testing.assert_array_equal(
        domains.id,
        np.zeros(5),
    )
    np.testing.assert_array_equal(
        domains.unq,
        np.arange(5),
    )


def test_find_domains_with_duplicated_interfaces():
    zeta = np.array([
        0.00,
        0.25,
        0.50,
        0.50,
        0.75,
        1.00,
        1.00,
        1.50,
        2.00,
    ])

    domains = find_domains(zeta)

    assert domains.Nd == 3

    np.testing.assert_allclose(
        domains.bounds,
        np.array([0.50, 1.00]),
        rtol=0.0,
        atol=0.0,
    )

    # First copy: end of the lower domain.
    np.testing.assert_array_equal(
        domains.end,
        np.array([2, 5]),
    )

    # Second copy: beginning of the upper domain.
    np.testing.assert_array_equal(
        domains.beg,
        np.array([3, 6]),
    )

    np.testing.assert_array_equal(
        domains.edges,
        np.array([0, 3, 6, 9]),
    )
    np.testing.assert_array_equal(
        domains.sizes,
        np.array([3, 3, 3]),
    )
    np.testing.assert_array_equal(
        domains.id,
        np.array([
            0, 0, 0,
            1, 1, 1,
            2, 2, 2,
        ]),
    )

    # One representative index is retained for each physical
    # coordinate, using its first occurrence.
    np.testing.assert_array_equal(
        domains.unq,
        np.array([0, 1, 2, 4, 5, 7, 8]),
    )

    np.testing.assert_array_equal(
        domains.int,
        np.array([
            True, True, True,
            True, True, True,
            False, False, False,
        ]),
    )
    np.testing.assert_array_equal(
        domains.ext,
        ~domains.int,
    )
    
    
def test_init_phi_c_dispatches_rotation_parameters():
    r = np.linspace(0.0, 1.0, 11)
    cos_theta = 0.37
    omega = 0.63
    alpha = 0.4
    scale = 0.25

    cases = [
        (solid, ()),
        (lorentzian, (alpha,)),
        (plateau, (alpha, scale)),
    ]

    for rotation_profile, profile_args in cases:
        potential_function, profile_function = init_phi_c(
            rotation_profile,
            central_diff_rate=alpha,
            rotation_scale=scale,
        )

        potential, derivative = potential_function(
            r,
            cos_theta,
            omega,
        )
        profile = profile_function(
            r,
            cos_theta,
            omega,
        )

        expected_potential, expected_derivative = rotation_profile(
            r,
            cos_theta,
            omega,
            *profile_args,
        )
        expected_profile = rotation_profile(
            r,
            cos_theta,
            omega,
            *profile_args,
            return_profile=True,
        )

        np.testing.assert_array_equal(
            potential,
            expected_potential,
        )
        np.testing.assert_array_equal(
            derivative,
            expected_derivative,
        )
        np.testing.assert_array_equal(
            profile,
            expected_profile,
        )
        
        
def test_valid_reciprocal_domain_for_monotonic_function():
    x = np.linspace(0.0, 1.0, 6)
    derivative = np.array([
        0.0,
        0.8,
        0.7,
        0.6,
        0.5,
        0.4,
    ])

    valid = valid_reciprocal_domain(x, derivative)

    np.testing.assert_array_equal(
        valid,
        np.array([
            False,
            True,
            True,
            True,
            True,
            True,
        ]),
    )


def test_valid_reciprocal_domain_stops_at_each_turning_point():
    x = np.linspace(0.0, 1.0, 6)

    derivative = np.array([
        [0.0, 0.0],
        [0.8, 0.9],
        [0.5, 0.4],
        [0.2, 0.0],
        [-0.1, -0.2],
        [-0.3, -0.4],
    ])

    valid = valid_reciprocal_domain(
        x,
        derivative,
        safety=1.0e-4,
    )

    expected = np.array([
        [False, False],
        [True,  True ],
        [True,  True ],
        [True,  False],
        [False, False],
        [False, False],
    ])

    np.testing.assert_array_equal(valid, expected)