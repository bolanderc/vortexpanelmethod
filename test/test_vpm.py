import numpy as np
from panelmethods import geometry_2D
from panelmethods import vortexpanelmethod


def test_2412_zero_alpha():
    mygeometry = geometry_2D.naca4digit("2412")
    naca2412 = vortexpanelmethod.VortexPanelMethod(mygeometry)
    cl_test, cm_test = naca2412.solve(aoa=0., v_mag=1.)

    cl = 0.25091555
    cm = -0.11862063

    assert np.allclose(cl_test, cl, rtol=0., atol=1e-7)
    assert np.allclose(cm_test, cm, rtol=0., atol=1e-7)
