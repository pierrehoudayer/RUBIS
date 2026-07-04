import numpy as np

from rubis.interpolation import interpolate_func


def test_interpolate_func_reproduces_cubic_and_derivative():
    x = np.linspace(0.0, 1.0, 17)**2

    def function(x):
        return 1 - 2*x + 3*x**2 - 0.5*x**3

    def derivative(x):
        return -2 + 6*x - 1.5*x**2

    interpolant = interpolate_func(
        x,
        function(x),
        der=(0, 1),
        k=3,
    )

    x_eval = np.linspace(0.0, 1.0, 41)
    values, derivatives = interpolant(x_eval)

    np.testing.assert_allclose(
        values,
        function(x_eval),
        rtol=1.0e-12,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        derivatives,
        derivative(x_eval),
        rtol=1.0e-11,
        atol=1.0e-12,
    )
    
    
def test_interpolate_func_applies_primitive_condition():
    x = np.linspace(0.0, 1.0, 17)**2

    def function(x):
        return 1 - 2*x + 3*x**2

    def primitive(x):
        return x - x**2 + x**3

    index = 6
    value = 2.3

    antiderivative = interpolate_func(
        x,
        function(x),
        der=-1,
        k=3,
        prim_cond=(index, value),
    )

    x_eval = np.linspace(0.0, 1.0, 41)
    expected = (
        primitive(x_eval)
        - primitive(x[index])
        + value
    )

    np.testing.assert_allclose(
        antiderivative(x_eval),
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    
    
def test_interpolate_func_preserves_empty_evaluations():
    x = np.linspace(0.0, 1.0, 7)
    y = x**2
    empty = np.array([])

    value = interpolate_func(
        x,
        y,
        der=0,
    )(empty)
    values = interpolate_func(
        x,
        y,
        der=(0, 1),
    )(empty)

    assert value.shape == (0,)
    assert values.shape == (2, 0)