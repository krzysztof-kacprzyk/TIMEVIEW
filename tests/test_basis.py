import pytest
from tts.basis import BSplineBasis
import numpy as np
import matplotlib.pyplot as plt

@pytest.fixture
def bspline_fixture():
    return BSplineBasis(7, (0, 1))

def test_get_knots(bspline_fixture):
    assert np.array_equal(bspline_fixture.get_knots(),np.array([0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1]))


def test_basis_decomposition(bspline_fixture):
    rng = np.random.default_rng(0)
    coeffs = rng.random(bspline_fixture.n_basis)
    spline = bspline_fixture.get_spline_with_coeffs(coeffs)
    t = np.linspace(0, 1, 100)
    result = np.zeros(100)

    for i in range(bspline_fixture.n_basis):
        basis = bspline_fixture.get_basis(i)
        result += coeffs[i] * basis(t)
        
    assert np.array_equal(spline(t), result)

def test_monomial_basis(bspline_fixture):
    internal_knots = bspline_fixture.internal_knots
    n_basis = bspline_fixture.n_basis
    for i in range(n_basis):
        basis = bspline_fixture.get_basis(i)
        monomial_matrix = bspline_fixture._monomial_basis(i)
        for j in range(len(internal_knots)-1):
            t = np.linspace(internal_knots[j], internal_knots[j+1], 10)
            result = monomial_matrix[0,j] + t * monomial_matrix[1,j] + t**2 * monomial_matrix[2,j] + t**3 * monomial_matrix[3,j]
            assert np.allclose(basis(t), result, atol=1e-10)

def test_spline_with_coeffs_monomial(bspline_fixture):
    internal_knots = bspline_fixture.internal_knots
    n_basis = bspline_fixture.n_basis
    rng = np.random.default_rng(0)
    coeffs = rng.random(n_basis)
    in_monomial = bspline_fixture.get_spline_with_coeffs_monomial(coeffs)
    spline = bspline_fixture.get_spline_with_coeffs(coeffs)
    for j in range(len(internal_knots)-1):
        t = np.linspace(internal_knots[j], internal_knots[j+1], 10)
        result = in_monomial[0,j] + t * in_monomial[1,j] + t**2 * in_monomial[2,j] + t**3 * in_monomial[3,j]
        assert np.allclose(spline(t), result, atol=1e-10)

@pytest.mark.parametrize("cubic, interval, template, points", [
    # Linear
    ([0,1,0,0], [-1,1], [0], [-1,1]),
    ([0,-1,0,0], [-1,1], [1], [-1,1]),
    ([0,0,0,0], [-1,1], [2], [-1,1]),
    # Convex parabola
    ([0,0,1,0], [-1,1], [5,3], [-1,0,1]),
    ([0,0,1,0], [0,1], [3], [0,1]),
    ([0,0,1,0], [0.5,1], [3], [0.5,1]),
    ([0,0,1,0], [-1,0], [5], [-1,0]),
    ([0,0,1,0], [-1,-0.5], [5], [-1,-0.5]),
    # Concave parabola
    ([0,0,-1,0], [-1,1], [4,6], [-1,0,1]),
    ([0,0,-1,0], [0,1], [6], [0,1]),
    ([0,0,-1,0], [0.5,1], [6], [0.5,1]),
    ([0,0,-1,0], [-1,0], [4], [-1,0]),
    ([0,0,-1,0], [-1,-0.5], [4], [-1,-0.5]),
    # Cubic
    ([0,0,0,1], [-1,1], [4,3], [-1,0,1]),
    ([0,0,0,1], [0,1], [3], [0,1]),
    ([0,0,0,1], [-1,0], [4], [-1,0]),
    ([0,0,0,-1], [-1,1], [5,6], [-1,0,1]),

    ([0,-3,0,1], [-2,2], [4,6,5,3], [-2,-1,0,1,2]),
    ([0,-3,0,1], [-2,1], [4,6,5], [-2,-1,0,1]),
    ([0,-3,0,1], [-2,0], [4,6], [-2,-1,0]),
    ([0,-3,0,1], [-2,-1], [4], [-2,-1]),
    ([0,-3,0,1], [-1,2], [6,5,3], [-1,0,1,2]),
    ([0,-3,0,1], [-1,1], [6,5], [-1,0,1]),
    ([0,-3,0,1], [-1,0], [6], [-1,0]),
    ([0,-3,0,1], [0,2], [5,3], [0,1,2]),
    ([0,-3,0,1], [0,1], [5], [0,1]),
    ([0,-3,0,1], [1,2], [3], [1,2])
])

def test_get_template_from_cubic(cubic, interval, template, points):
    bs = BSplineBasis(5, interval)
    found_template, found_points = bs.get_template_from_cubic(np.array(cubic), interval)
    assert np.array_equal(found_template, template)
    assert np.isclose(found_points, points).all()

def test_get_template_from_coeffs():
    bs = BSplineBasis(10, (0, 1))
    rng = np.random.default_rng(0)
    coeffs = rng.random(bs.n_basis)
    assert bs.get_template_from_coeffs(coeffs)[0] == [5,3,4,6,5,3,4,6,5,3]

