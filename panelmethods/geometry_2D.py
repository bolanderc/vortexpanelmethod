"""Implements 2D geometries for use in panel methods."""

import numpy as np


def naca4digit(NACAnum, num_nodes=240):
    """
    Generates points for NACA 4-digit series airfoils.

    Takes the number of nodes as well as the NACA number and
    finds the x and y coordinates of the nodes for a given airfoil.
    """
    #Initialization of parameters
    num_panels = num_nodes - 1
    ycamt = 0.0
    ycamb = 0.0
    xn = np.zeros(num_nodes)
    x = np.zeros(num_nodes)
    y = np.zeros(num_nodes)
    xc = np.zeros(num_panels)
    yc = np.zeros(num_panels)
    n = int(num_nodes)
    ytu = 0.0
    ytl = 0.0
    #Translates NACA number into corresponding info for thickness and camber
    m = float(NACAnum[0])/100
    p = float(NACAnum[1])/10
    th = float(NACAnum[2:4])/100
    #Cosine clustering of airfoil 
    dtheta = (np.pi*2)/(num_nodes - 1)
    #Coefficients from Wikipedia page on NACA airfoils to find the thickness
    yt_c = [0.2969,0.126,0.3516,0.2843,0.1036]
    for i in range(1,n//2+1,1):
        #The cosine clustered x values are calculated according to Eq(1.6.17)
        #in Phillips' text. xtb(bottom) and xtt(top are archived for easier use
        #later.
        xn[n//2 +1-i-1] = 0.5*(1 -np.cos((i-0.5)*dtheta))
        xtb = xn[n//2 +1-i-1]
        xn[n//2 + i-1] = 0.5*(1 -np.cos((i-0.5)*dtheta))
        xtt = xn[n//2 + i-1]
        #Calculates thickness of airfoil top and bottom according to wikipedia
        #equations
        ytu = ((5*th)*((yt_c[0]*np.sqrt(xtt))-(yt_c[1]*xtt) - (yt_c[2]*(xtt**2))
        + (yt_c[3]*(xtt**3)) - (yt_c[4]*(xtt**4))))
        ytl = ((5*th)*((yt_c[0]*np.sqrt(xtb))-(yt_c[1]*xtb) - (yt_c[2]*(xtb**2))
        + (yt_c[3]*(xtb**3)) - (yt_c[4]*(xtb**4))))
        #Wikipedia equations to find the y camber line as well as its slope
        if 0<=xtt<=p:
            ycamt = (m/(p**2))*((2*p)*(xtt) - (xtt**2))
            dyc_dxu = ((2*m)/(p**2))*(p-xtt)
        elif p<xtt<=1:
            ycamt = (m/((1-p)**2))*((1-2*p)+(2*p)*(xtt) -(xtt**2))
            dyc_dxu = ((2*m)/((1-p)**2))*(p-xtt)
        if 0<=xtb<=p:
            ycamb = (m/(p**2))*((2*p)*(xtb) - (xtb**2))
            dyc_dxb = ((2*m)/(p**2))*(p-xtb)
        elif p<xtb<=1:
            ycamb = (m/((1-p)**2))*((1-2*p)+(2*p)*(xtb) -(xtb**2))
            dyc_dxb = ((2*m)/((1-p)**2))*(p-xtb)
        #Theta on upper and lower surfaces calculated to find final x and y
        #values for the airfoil.
        thetau = np.arctan(dyc_dxu)
        thetal = np.arctan(dyc_dxb)
        x[n//2 +1-i-1] = xtt - ytu*np.sin(thetau)
        x[n//2+i-1] = xtb +ytl*np.sin(thetal)
        y[n//2 +1-i-1] = ycamt - ytu*np.cos(thetau)
        y[n//2+i-1] = ycamb +ytl*np.cos(thetal)
    #Finds control point coordinates for each panel connecting the nodes.
    for i in range(n-1):
        xc[i] = (x[i] + x[i+1])/2
        yc[i] = (y[i] + y[i+1])/2
    #Graphs the shape of the airfoil and returns x,y,xc,and yc.
    # plt.figure(0)
    # plt.title('Airfoil Shape')
    # plt.plot(x,y)
    # plt.xlabel('Chord')
    # plt.ylabel('Thickness')
    # plt.ylim(-0.3,0.3)
    return np.array([x, y]).T
