# -*- coding: utf-8 -*-
"""A 2D vortex panel method implementation for inviscid flow.

This module implements a 2D vortex panel method for inviscid flow. As the code
is contained in a class structure, it is easy to sweep through velocities and
angles of attack with a single instance of the class, as the A matrix will be
saved the first time that it is run. The lift and moment coefficients are
returned for a given geometry as outputs.

Routine Listings
-----------------
VortexPanelMethod : Class for storing the elements required for the vortex
                    panel method as well as solving for the vortex
                    strengths and aerodynamic coefficients.

Notes
-------
The moment coefficient calculated here is the moment coefficient about the
leading edge.

References
-----------
Phillips, Warren F. Mechanics of flight. John Wiley & Sons, 2004.

Example
--------
from panelmethods import geometry_2D
from panelmethods import vortexpanelmethod

mygeometry = geometry_2D.naca4digit("2412")
naca2412 = vortexpanelmethod.VortexPanelMethod(mygeometry)
cl, cm = naca2412.solve(aoa=0., v_mag=1.)

print("Lift Coeffecient: ", cl)
print("Moment Coeffecient: ", cm)

"""
import numpy as np


class VortexPanelMethod:
    """VortexPanelMethod is able to use a series of `x` and `y` data points to
    calculate aerodynamic coefficients.

    This class houses the functions that allow the vortex panel method to be
    applied to a given geometry to find the lift and moment coefficients. This
    class is meant to be a tool that can be easily combined with other programs
    in this module to perform calculations on inviscid flow over objects.

    The __init__ constructor will run each time an instance of the class is
    created. It inherits from the geometry created by `geometry_2D.py` and will
    solve for the matrix in the vortex panel method that is on the same side as
    the nodal strengths. This value will be stored and accessible in that
    of the class to increase the efficiency of the algorithm when sweeping
    through angles of attack or velocities.

    Attributes
    ------------
    geometry : array_like
        Contains an array of `x` and `y` coordinate values.
    _num_nodes : int
        The number of nodes over the geometry.
    _A : array
        The airfoil coefficient matrix used to solve for the nodal strengths
        of the geometry.
    _b : array
        Velocity terms used to solve for the nodal strengths of the geometry.
    _results : tuple
        Contains the section lift coefficient and moment coefficient about the
        leading edge.

    See Also
    -----------
    numpy.dot : Used for matrix multiplication or dot products

    numpy.arctan2 : Calculates the four-quadrant arctangent

    numpy.linalg.solve : Solves a matrix equation of the form `Ax=b` for `x`

    geometry_2D.py : Returns `x` and `y` data points for a given NACA airfoil

    """
    def __init__(self, geometry):
        self._geometry = geometry
        self._num_nodes = len(geometry)
#        self._P = self._calc_panel_coeff()
        self._A = self._calc_airfoil_coeff()
        self._b = None
        self._results = None

    def _calc_airfoil_coeff(self):
        n = self._num_nodes
        x, y = self._geometry.T
        x = x/np.sqrt(1 - 100*100/(343*343))
        x_c = (x[:-1] + x[1:])/2.
        y_c = (y[:-1] + y[1:])/2.
        # Eq. 1.6.19 in text calculates the length of each panel
        self._pan_len = np.sqrt((x[1:]-x[:-1])**2 + (y[1:]-y[:-1])**2)
        panel_mat_a = np.zeros((2, 2))
        panel_mat_b = np.zeros((2, 2))
        panel_tangent = 0.0
        panel_normal = 0.0
        panel_lengths = np.zeros((2, 2))
        panel_cp_dist = np.zeros((2, 1))
        panel_mat = np.zeros((2, 2))
        airfoil_mat = np.zeros((n, n))
        for i in range(n-1):
            for j in range(n-1):
                # panel_lengths and panel_cp_dist are used as the 2x2 matrix
                # (panel_lengths) and the 2x1 matrix(panel_cp_dist) in equation
                # 1.6.20 in Phillips' text.
                panel_lengths[0, 0] = (x[j+1] - x[j])
                panel_lengths[0, 1] = (y[j+1] - y[j])
                panel_lengths[1, 0] = -(y[j+1] - y[j])
                panel_lengths[1, 1] = (x[j+1] - x[j])
                panel_cp_dist[0, 0] = (x_c[i] - x[j])
                panel_cp_dist[1, 0] = (y_c[i] - y[j])
                # Eq. 1.6.20 returns the panel coordinate system.
                panel_coords = (np.dot(panel_lengths,
                                       panel_cp_dist)/self._pan_len[j])
                panel_tangent = panel_coords[0]
                panel_normal = panel_coords[1]
                # Eq. 1.6.21 returns phi and Eq. 1.6.22 returns psi.
                phi = np.arctan2(panel_normal*self._pan_len[j],
                                 (panel_normal**2 + panel_tangent**2 -
                                  panel_tangent*self._pan_len[j]))
                psi = (0.5*np.log((panel_tangent**2 + panel_normal**2) /
                                  ((panel_tangent-self._pan_len[j])**2 +
                                   panel_normal**2)))
                # Eq. 1.6.23 calculates panel coefficient matrix, used later to
                # find the airfoil coefficient matrix.
                panel_mat_a[0, 0] = (x[j+1] - x[j])
                panel_mat_a[0, 1] = -(y[j+1] - y[j])
                panel_mat_a[1, 0] = (y[j+1] - y[j])
                panel_mat_a[1, 1] = (x[j+1] - x[j])
                panel_mat_b[0, 0] = ((self._pan_len[j] - panel_tangent)*phi +
                                     panel_normal*psi)
                panel_mat_b[0, 1] = (panel_tangent*phi - panel_normal*psi)
                panel_mat_b[1, 0] = (panel_normal*phi -
                                     (self._pan_len[j] - panel_tangent)*psi -
                                     self._pan_len[j])
                panel_mat_b[1, 1] = (-panel_normal*phi - panel_tangent*psi +
                                     self._pan_len[j])
                panel_mat = (np.dot(panel_mat_a, panel_mat_b) /
                             (2*np.pi*(self._pan_len[j]**2)))
                # Uses above data to calculate the airfoil coefficient matrix.
                # Physically represents the velocity induced at control
                # point i by panel j.
                airfoil_mat[i, j] = (airfoil_mat[i, j] +
                                     (((x[i+1] - x[i])/self._pan_len[i]) *
                                      (panel_mat[1, 0])) -
                                     (((y[i+1] - y[i]) / self._pan_len[i]) *
                                      (panel_mat[0, 0])))
                airfoil_mat[i, j+1] = (airfoil_mat[i, j+1] +
                                       (((x[i+1] - x[i])/self._pan_len[i]) *
                                        (panel_mat[1, 1])) -
                                       (((y[i+1] - y[i])/self._pan_len[i]) *
                                        (panel_mat[0, 1])))
        # Enforces the Kutta condition
        airfoil_mat[n-1, 0] = 1.0
        airfoil_mat[n-1, n-1] = 1.0
        return airfoil_mat

    def solve(self, aoa, v_mag, comp_corr=''):
        """Solves for the nodal vortex strengths (gamma) of the geometry.

        Utilizes a linear algebra solver to obtain values for the lift and
        moment coefficients given the airfoil coefficient matrix and the
        velocity terms in vector form.

        Returns
        -------
        _results : tuple
            Contains the resulting section lift and moment coefficients

        """
        self.m_inf = v_mag/343
        self._b = self._calc_freestream_terms(aoa, v_mag)
        self._gamma = np.linalg.solve(self._A, self._b)
        self._results = self._calc_forces_moments(aoa, v_mag, comp_corr)

        return self._results

    def _calc_freestream_terms(self, aoa, v_mag):
        x, y = self._geometry.T
        n = self._num_nodes
        freestream_terms = np.zeros((n, 1))
        # Generates the vector with the freestream terms.
        for i in range(n-1):
            freestream_terms[i] = ((((y[i+1] - y[i])*np.cos(aoa*(np.pi/180))) -
                                   ((x[i+1] - x[i])*np.sin(aoa*(np.pi/180)))) /
                                   self._pan_len[i])
        # Kutta Condition
        freestream_terms[n-1] = 0.0
        return freestream_terms

    def _calc_forces_moments(self, aoa, v_mag, comp_corr):
        M = self.m_inf
        gamma = self._gamma
        pan_len = self._pan_len
        x, y = self._geometry.T
        n = self._num_nodes
        c_l = 0.0
        c_m_le = 0.0
        # Eq. 1.6.32 to find C_l and 1.6.33 to find C_mle
        for i in range(n-1):
            c_l += (pan_len[i]*((gamma[i] + gamma[i+1])/v_mag))
            cm1 = ((2*x[i]*gamma[i] + x[i]*gamma[i+1] + x[i+1]*gamma[i] +
                    2*x[i+1]*gamma[i+1])/v_mag)
            cm2 = ((2*y[i]*gamma[i] + y[i]*gamma[i+1] + y[i+1]*gamma[i] +
                    2*y[i+1]*gamma[i+1])/v_mag)
            c_m_le += (pan_len[i]*(cm1*np.cos(aoa*(np.pi/180)) +
                       cm2*np.sin(aoa*(np.pi/180))))
        c_m_le = c_m_le*(-1.0/3.0)
        cl = np.sum(c_l)
        if comp_corr == 'PG':
            correction = 1./np.sqrt(1 - M*M)
            return cl*correction, c_m_le*correction
        elif comp_corr == 'L':
            D1 = np.sqrt(1. - M*M)
            D2 = (M*M*(1. + 0.2*M*M))/(2*D1)
            correction = 1./(D1 + D2*cl)
            return cl*correction, c_m_le*correction
        elif comp_corr == 'KT':
            D1 = np.sqrt(1 - M*M)
            D2 = (M*M)/(1. + D1)
            correction = 1./(D1 + D2*cl/2.)
            return cl*correction, c_m_le*correction
        else:
            return cl, c_m_le
