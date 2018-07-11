"""
Christian Bolander
MAE 5500 Aerodynamics Project 3
Vortex Sheet Method
"""
import numpy as np
import matplotlib.pyplot as plt
import tabulate as tb


"""
Find Geometry takes the number of nodes as well as the NACA number and
finds the x and y coordinates of the nodes for a given airfoil. It also finds
the locations of the control points on the airfoil.
"""
def FindGeometry(num_nodes,NACAnum):
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
    m = NACAnum[0]/100
    p = NACAnum[1]/10
    th = NACAnum[2]/100
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
    plt.figure(0)
    plt.title('Airfoil Shape')
    plt.plot(x,y)
    plt.xlabel('Chord')
    plt.ylabel('Thickness')
    plt.ylim(-0.3,0.3)
    return x,y,xc,yc
"""
Does the necessary calculations for the Vortex Method. Takes the number of nodes,
the NACA airfoil number, freestream velocity, and angle of attack. Returns Lift
Coefficient and Moment Coefficient on the leading edge. Equations 1.6.19 - 
1.6.33 are used, excluding 1.6.19 - 1.6.31.
"""
def VortexMethod(num_nodes,NACAnum,V_inf,alpha):
    #Initialization of parameters
    n = num_nodes
    x,y,xc,yc = FindGeometry(n,NACAnum)
    a = np.zeros((2,2))
    b = np.zeros((2,2))
    l = np.zeros(n-1)
    zeta = 0.0
    eta = 0.0
    Cl = 0.0
    Cmle = 0.0
    za = np.zeros((2,2))
    zb = np.zeros((2,1))
    P = np.zeros((2,2))
    A = np.zeros((n,n))
    B = np.zeros((n,1))
    gamma = np.zeros(n)
    #Constructs an A and B matrix to calculate the strength of each of the 
    #vorticies at each node.
    for i in range(n-1):
        for j in range(n-1):
            #za and ab are used as the 2x2 matrix (za) and the 2x1 matrix(zb) 
            #in equation 1.6.20 in Phillips' text.
            za[0][0] = (x[j+1]-x[j])
            za[0][1] = (y[j+1]-y[j])
            za[1][0] = -(y[j+1] - y[j])
            za[1][1] = (x[j+1] - x[j])
            zb[0][0] = (xc[i] - x[j])
            zb[1][0] = (yc[i] - y[j])
            #Eq. 1.6.19 in text calculates the length of each panel
            l[j] = np.sqrt((x[j+1]-x[j])**2 + (y[j+1]-y[j])**2)
            #Eq. 1.6.20, returns zeta and eta values used later.
            intermediate = (np.dot(za,zb)/l[j])
            zeta = intermediate[0]
            eta = intermediate[1]
            #Eq. 1.6.21 returns Phi and Eq. 1.6.22 returns Psi, used later to
            #find P matrix.
            Phi = np.arctan2(eta*l[j],(eta**2 + zeta**2 - zeta*l[j]))
            Psi = 0.5*np.log((zeta**2 + eta**2)/((zeta-l[j])**2 + eta**2))
            #Eq. 1.6.23 calculates P matrix, used later to find A matrix. Called
            #panel coefficient matrix.
            a[0][0] = (x[j+1] - x[j])
            a[0][1] = -(y[j+1] - y[j])
            a[1][0] = (y[j+1] - y[j])
            a[1][1] = (x[j+1] - x[j])
            b[0][0] = ((l[j] - zeta)*Phi + eta*Psi)
            b[0][1] = (zeta*Phi - eta*Psi)
            b[1][0] = (eta*Phi - (l[j] - zeta)*Psi - l[j])
            b[1][1] = (-eta*Phi - zeta*Psi + l[j])
            P = np.dot(a,b)/(2*np.pi*(l[j]**2))
            #Uses above data to calculate the A matrix (airfoil coefficient
            #matrix). Physically represents the velocity induced at control
            #point i by panel j.
            A[i][j] = (A[i][j] + (((x[i+1] - x[i])/l[i])*(P[1][0])) - (((y[i+1]
            - y[i])/l[i])*(P[0][0])))
            A[i][j+1] = (A[i][j+1] + (((x[i+1] - x[i])/l[i])*(P[1][1])) - 
             (((y[i+1] - y[i])/l[i])*(P[0][1])))
    A[n-1][0] = 1.0
    A[n-1][n-1] = 1.0
    #Generates the B matrix, which represents free stream velocity
    #considerations.
    for i in range(n-1):
        B[i] = ((((y[i+1] - y[i])*np.cos(alpha*(np.pi/180))) - 
         ((x[i+1] - x[i])*np.sin(alpha*(np.pi/180))))/l[i])
    #Kutta Condition
    B[n-1] = 0.0
    #Nodal vortex strengths
    gamma = np.linalg.solve(A,B)
    #Eq. 1.6.32 to find C_l and 1.6.33 to find C_mle
    for i in range(n-1):
        Cl += (l[i]*((gamma[i] + gamma[i+1])/V_inf))
        Cm1 = ((2*x[i]*gamma[i] + x[i]*gamma[i+1] + x[i+1]*gamma[i] + 
                2*x[i+1]*gamma[i+1])/V_inf)
        Cm2 = ((2*y[i]*gamma[i] + y[i]*gamma[i+1] + y[i+1]*gamma[i] + 
                2*y[i+1]*gamma[i+1])/V_inf)
        Cmle += (l[i]*(Cm1*np.cos(alpha*(np.pi/180)) + 
                 Cm2*np.sin(alpha*(np.pi/180))))
    Cmle = Cmle*(-1.0/3.0)
    return Cl,Cmle
"""
Specific to Aerodynamics project. Tabulates and plots the coefficients 
calculated above as a function of angle of attack and compares values returned
by Vortex Panel Method to values obtained experimentally recorded in Figure 
4.37 in Anderson's Aerodynamics text. 
"""
def Plot_and_Tab(num_nodes,NACAnum,V_inf):
    i = 0
    #Alpha range given in project
    alpha_range = np.arange(-8,16,4)
    #Experimental values from Figure 4.37
    Clexp = [-0.6,-0.2,0.23,0.65,1.09,1.42]
    Cmlexp = [0.104,0.008,-0.0965,-0.1975,-0.3055,-0.383]
    Cl = np.zeros(len(alpha_range))
    Cmle = np.zeros(len(alpha_range))
    #Vortex Panel Method values for comparison
    for a in alpha_range:    
        Cl[i], Cmle[i] = VortexMethod(num_nodes,NACAnum,V_inf,a)
        i+=1
    #Plots and Tables
    plt.figure(1)
    plt.title('Lift Coefficient Comparison')
    plt.scatter(alpha_range,Clexp, label = 'Experimental')
    plt.scatter(alpha_range,Cl, label = 'Vortex Method Approximation')
    plt.legend(loc = 'upper left')
    plt.xlabel('Angle of Attack (degrees)')
    plt.ylabel('Coefficient of Lift (C_l)')
    table = [[0 for x in range(len(alpha_range))]for y in range(len(alpha_range))] 
    for i in range(len(alpha_range)):
        table[i] = (alpha_range[i],Cl[i],Clexp[i])
    headers = ["Alpha","C_L Vortex Panel","C_L Experimental"]
    t = tb.tabulate(table,headers,floatfmt=".6f")
    print(t)           
    plt.figure(2)
    plt.title('Moment Coefficient Comparison')
    plt.scatter(alpha_range,Cmlexp, label = 'Experimental')
    plt.scatter(alpha_range,Cmle, label = 'Vortex Method Approximation')
    plt.legend(loc = 'upper right')
    plt.xlabel('Angle of Attack (degrees)')
    plt.ylabel('Moment Coefficient (About Leading Edge)')
    table = [[0 for x in range(len(alpha_range))]for y in range(len(alpha_range))] 
    for i in range(len(alpha_range)):
        table[i] = (alpha_range[i],Cmle[i],Cmlexp[i])
    headers = ["Alpha","C_mLE Vortex Panel","C_mLE Experimental"]
    t = tb.tabulate(table,headers,floatfmt=".6f")
    print(t)       
Plot_and_Tab(30,[0,0,12],1)           
            
            
            
            