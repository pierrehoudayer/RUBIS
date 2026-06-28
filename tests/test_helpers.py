import numpy as np

from rubis.domains import DomainLayout, find_domains
from rubis.mapping import valid_reciprocal_domain


def test_find_domains_for_continuous_coordinate():
    zeta = np.linspace(0.0, 1.0, 5)

    domains = find_domains(zeta)

    assert domains.n_domains == 1
    assert isinstance(domains, DomainLayout)
    assert not domains.has_interfaces
    assert domains.interface_indices == ()
    
    np.testing.assert_array_equal(
        domains.interface_end_indices,
        np.empty(0, dtype=int),
    )
    np.testing.assert_array_equal(
        domains.interface_start_indices,
        np.empty(0, dtype=int),
    )
    np.testing.assert_array_equal(
        domains.interface_values,
        np.array([]),
    )
    np.testing.assert_array_equal(
        domains.domain_edges,
        np.array([0, 5]),
    )
    np.testing.assert_array_equal(
        domains.domain_sizes,
        np.array([5]),
    )
    np.testing.assert_array_equal(
        domains.domain_index,
        np.zeros(5),
    )
    np.testing.assert_array_equal(
        domains.unique_indices,
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

    assert domains.n_domains == 3
    assert isinstance(domains, DomainLayout)

    np.testing.assert_allclose(
        domains.interface_values,
        np.array([0.50, 1.00]),
        rtol=0.0,
        atol=0.0,
    )

    # First copy: end of the lower domain.
    np.testing.assert_array_equal(
        domains.interface_end_indices,
        np.array([2, 5]),
    )

    # Second copy: beginning of the upper domain.
    np.testing.assert_array_equal(
        domains.interface_start_indices,
        np.array([3, 6]),
    )

    np.testing.assert_array_equal(
        domains.domain_edges,
        np.array([0, 3, 6, 9]),
    )
    np.testing.assert_array_equal(
        domains.domain_sizes,
        np.array([3, 3, 3]),
    )
    np.testing.assert_array_equal(
        domains.domain_index,
        np.array([
            0, 0, 0,
            1, 1, 1,
            2, 2, 2,
        ]),
    )

    # One representative index is retained for each physical
    # coordinate, using its first occurrence.
    np.testing.assert_array_equal(
        domains.unique_indices,
        np.array([0, 1, 2, 4, 5, 7, 8]),
    )

    np.testing.assert_array_equal(
        domains.internal_mask,
        np.array([
            True, True, True,
            True, True, True,
            False, False, False,
        ]),
    )
    np.testing.assert_array_equal(
        domains.external_mask,
        ~domains.internal_mask,
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