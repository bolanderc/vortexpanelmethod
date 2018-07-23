from panelmethods import geometry_2D
import numpy as np


def test_naca4digit():
    test_geom = geometry_2D.naca4digit("0012", 6)

    geom = np.array([[1.00000000e+00, 1.66533454e-17],
                     [6.54508497e-01, -4.06861839e-02],
                     [9.54915028e-02, -4.60488284e-02],
                     [9.54915028e-02, 4.60488284e-02],
                     [6.54508497e-01, 4.06861839e-02],
                     [1.00000000e+00, -1.66533454e-17]])

    assert np.allclose(test_geom, geom, rtol=0., atol=1e-9)
