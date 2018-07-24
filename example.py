from panelmethods import geometry_2D
from panelmethods import vortexpanelmethod

mygeometry = geometry_2D.naca4digit("2412")
naca2412 = vortexpanelmethod.VortexPanelMethod(mygeometry)
cl, cm = naca2412.solve(aoa=0., v_mag=1.)

print("Lift Coeffecient: ", cl)
print("Moment Coeffecient: ", cm)


# arbitrary geometries
# sweep through multiple angles of attack and/or velocities
# multiple airfoils/objects
