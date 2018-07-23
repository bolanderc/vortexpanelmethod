from panelmethods import geometry_2D
from panelmethods import VortexPanelMethod

mygeometry = geometry_2D.naca("2412")
naca2412 = VortexPanelMethod(mygeometry)
results = naca2412.solve(aoa=1., v_mag=40.)

print("Lift Coeffecient: ", results["CL"])
print("Moment Coeffecient: ", results["Cm"])


# arbitrary geometries
# sweep through multiple angles of attack and/or velocities
# multiple airfoils/objects
