import pytest
from tts.basis import BSplineBasis
import numpy as np

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