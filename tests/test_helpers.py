import numpy as np

from helpers import find_domains


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