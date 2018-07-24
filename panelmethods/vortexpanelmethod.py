"""
Christian Bolander
MAE 5500 Aerodynamics Project 3
Vortex Sheet Method
"""
import numpy as np
import matplotlib.pyplot as plt


class VortexPanelMethod:
    """

    """
    def __init__(self, geometry):
        self._geometry = geometry
        self._n = len(geometry)
#        self._P = self._calc_panel_coeff()
        self._A = self._calc_airfoil_coeff()
        self._b = None
        self._results = None

    def _calc_airfoil_coeff(self):
        n = self._n
        x, y = self._geometry.T
        x_c = (x[:-1] + x[1:])/2.
        y_c = (y[:-1] + y[1:])/2.
        #Eq. 1.6.19 in text calculates the length of each panel
        self._l = np.sqrt((x[1:]-x[:-1])**2 + (y[1:]-y[:-1])**2)
        a = np.zeros((2,2))
        b = np.zeros((2,2))
        zeta = 0.0
        eta = 0.0
        za = np.zeros((2,2))
        zb = np.zeros((2,1))
        P = np.zeros((2,2))
        A = np.zeros((n,n))
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
                zb[0][0] = (x_c[i] - x[j])
                zb[1][0] = (y_c[i] - y[j])
                #Eq. 1.6.20, returns zeta and eta values used later.
                intermediate = (np.dot(za,zb)/self._l[j])
                zeta = intermediate[0]
                eta = intermediate[1]
                #Eq. 1.6.21 returns Phi and Eq. 1.6.22 returns Psi, used later to
                #find P matrix.
                Phi = np.arctan2(eta*self._l[j],(eta**2 + zeta**2 - zeta*self._l[j]))
                Psi = 0.5*np.log((zeta**2 + eta**2)/((zeta-self._l[j])**2 + eta**2))
                #Eq. 1.6.23 calculates P matrix, used later to find A matrix. Called
                #panel coefficient matrix.
                a[0][0] = (x[j+1] - x[j])
                a[0][1] = -(y[j+1] - y[j])
                a[1][0] = (y[j+1] - y[j])
                a[1][1] = (x[j+1] - x[j])
                b[0][0] = ((self._l[j] - zeta)*Phi + eta*Psi)
                b[0][1] = (zeta*Phi - eta*Psi)
                b[1][0] = (eta*Phi - (self._l[j] - zeta)*Psi - self._l[j])
                b[1][1] = (-eta*Phi - zeta*Psi + self._l[j])
                P = np.dot(a,b)/(2*np.pi*(self._l[j]**2))
                #Uses above data to calculate the A matrix (airfoil coefficient
                #matrix). Physically represents the velocity induced at control
                #point i by panel j.
                A[i][j] = (A[i][j] + (((x[i+1] - x[i])/self._l[i])*(P[1][0])) - (((y[i+1]
                - y[i])/self._l[i])*(P[0][0])))
                A[i][j+1] = (A[i][j+1] + (((x[i+1] - x[i])/self._l[i])*(P[1][1])) - 
                 (((y[i+1] - y[i])/self._l[i])*(P[0][1])))
        A[n-1][0] = 1.0
        A[n-1][n-1] = 1.0
        return A


#    def _calc_panel_coeff(self):
#        pass

    def solve(self, aoa, v_mag):
        self._b = self._calc_rhs(aoa, v_mag)
        self._gamma = np.linalg.solve(self._A, self._b)
        self._results = self._calc_forces_moments(aoa, v_mag)

        return self._results

    def _calc_rhs(self, aoa, v_mag):
        x, y = self._geometry.T
        n = self._n
        B = np.zeros((n,1))
        #Generates the B matrix, which represents free stream velocity
        #considerations.
        for i in range(n-1):
            B[i] = ((((y[i+1] - y[i])*np.cos(aoa*(np.pi/180))) - 
             ((x[i+1] - x[i])*np.sin(aoa*(np.pi/180))))/self._l[i])
        #Kutta Condition
        B[n-1] = 0.0
        return B
    
    def _calc_forces_moments(self, aoa, v_mag):
        gamma = self._gamma
        l = self._l
        x, y = self._geometry.T
        n = self._n
        Cl = 0.0
        Cmle = 0.0
        #Eq. 1.6.32 to find C_l and 1.6.33 to find C_mle
        for i in range(n-1):
            Cl += (l[i]*((gamma[i] + gamma[i+1])/v_mag))
            Cm1 = ((2*x[i]*gamma[i] + x[i]*gamma[i+1] + x[i+1]*gamma[i] + 
                    2*x[i+1]*gamma[i+1])/v_mag)
            Cm2 = ((2*y[i]*gamma[i] + y[i]*gamma[i+1] + y[i+1]*gamma[i] + 
                    2*y[i+1]*gamma[i+1])/v_mag)
            Cmle += (l[i]*(Cm1*np.cos(aoa*(np.pi/180)) + 
                     Cm2*np.sin(aoa*(np.pi/180))))
        Cmle = Cmle*(-1.0/3.0)
        return Cl, Cmle

"""
Does the necessary calculations for the Vortex Method. Takes the number of nodes,
the NACA airfoil number, freestream velocity, and angle of attack. Returns Lift
Coefficient and Moment Coefficient on the leading edge. Equations 1.6.19 - 
1.6.33 are used, excluding 1.6.19 - 1.6.31.
"""
#def VortexMethod(num_nodes,NACAnum,V_inf,alpha):
#    #Initialization of parameters
#    n = num_nodes
#    x,y,xc,yc = FindGeometry(n,NACAnum)
#    a = np.zeros((2,2))
#    b = np.zeros((2,2))
#    l = np.zeros(n-1)
#    zeta = 0.0
#    eta = 0.0
#    Cl = 0.0
#    Cmle = 0.0
#    za = np.zeros((2,2))
#    zb = np.zeros((2,1))
#    P = np.zeros((2,2))
#    A = np.zeros((n,n))
#    B = np.zeros((n,1))
#    gamma = np.zeros(n)
#    #Constructs an A and B matrix to calculate the strength of each of the 
#    #vorticies at each node.
#    for i in range(n-1):
#        for j in range(n-1):
#            #za and ab are used as the 2x2 matrix (za) and the 2x1 matrix(zb) 
#            #in equation 1.6.20 in Phillips' text.
#            za[0][0] = (x[j+1]-x[j])
#            za[0][1] = (y[j+1]-y[j])
#            za[1][0] = -(y[j+1] - y[j])
#            za[1][1] = (x[j+1] - x[j])
#            zb[0][0] = (xc[i] - x[j])
#            zb[1][0] = (yc[i] - y[j])
#            #Eq. 1.6.19 in text calculates the length of each panel
#            l[j] = np.sqrt((x[j+1]-x[j])**2 + (y[j+1]-y[j])**2)
#            #Eq. 1.6.20, returns zeta and eta values used later.
#            intermediate = (np.dot(za,zb)/l[j])
#            zeta = intermediate[0]
#            eta = intermediate[1]
#            #Eq. 1.6.21 returns Phi and Eq. 1.6.22 returns Psi, used later to
#            #find P matrix.
#            Phi = np.arctan2(eta*l[j],(eta**2 + zeta**2 - zeta*l[j]))
#            Psi = 0.5*np.log((zeta**2 + eta**2)/((zeta-l[j])**2 + eta**2))
#            #Eq. 1.6.23 calculates P matrix, used later to find A matrix. Called
#            #panel coefficient matrix.
#            a[0][0] = (x[j+1] - x[j])
#            a[0][1] = -(y[j+1] - y[j])
#            a[1][0] = (y[j+1] - y[j])
#            a[1][1] = (x[j+1] - x[j])
#            b[0][0] = ((l[j] - zeta)*Phi + eta*Psi)
#            b[0][1] = (zeta*Phi - eta*Psi)
#            b[1][0] = (eta*Phi - (l[j] - zeta)*Psi - l[j])
#            b[1][1] = (-eta*Phi - zeta*Psi + l[j])
#            P = np.dot(a,b)/(2*np.pi*(l[j]**2))
#            #Uses above data to calculate the A matrix (airfoil coefficient
#            #matrix). Physically represents the velocity induced at control
#            #point i by panel j.
#            A[i][j] = (A[i][j] + (((x[i+1] - x[i])/l[i])*(P[1][0])) - (((y[i+1]
#            - y[i])/l[i])*(P[0][0])))
#            A[i][j+1] = (A[i][j+1] + (((x[i+1] - x[i])/l[i])*(P[1][1])) - 
#             (((y[i+1] - y[i])/l[i])*(P[0][1])))
#    A[n-1][0] = 1.0
#    A[n-1][n-1] = 1.0
#    #Generates the B matrix, which represents free stream velocity
#    #considerations.
#    for i in range(n-1):
#        B[i] = ((((y[i+1] - y[i])*np.cos(alpha*(np.pi/180))) - 
#         ((x[i+1] - x[i])*np.sin(alpha*(np.pi/180))))/l[i])
#    #Kutta Condition
#    B[n-1] = 0.0
#    #Nodal vortex strengths
#    gamma = np.linalg.solve(A,B)
#    #Eq. 1.6.32 to find C_l and 1.6.33 to find C_mle
#    for i in range(n-1):
#        Cl += (l[i]*((gamma[i] + gamma[i+1])/V_inf))
#        Cm1 = ((2*x[i]*gamma[i] + x[i]*gamma[i+1] + x[i+1]*gamma[i] + 
#                2*x[i+1]*gamma[i+1])/V_inf)
#        Cm2 = ((2*y[i]*gamma[i] + y[i]*gamma[i+1] + y[i+1]*gamma[i] + 
#                2*y[i+1]*gamma[i+1])/V_inf)
#        Cmle += (l[i]*(Cm1*np.cos(alpha*(np.pi/180)) + 
#                 Cm2*np.sin(alpha*(np.pi/180))))
#    Cmle = Cmle*(-1.0/3.0)
#    return Cl,Cmle
#"""
#Specific to Aerodynamics project. Tabulates and plots the coefficients 
#calculated above as a function of angle of attack and compares values returned
#by Vortex Panel Method to values obtained experimentally recorded in Figure 
#4.37 in Anderson's Aerodynamics text. 
#"""
#def Plot_and_Tab(num_nodes,NACAnum,V_inf):
#    i = 0
#    #Alpha range given in project
#    alpha_range = np.arange(-8,16,4)
#    #Experimental values from Figure 4.37
#    Clexp = [-0.6,-0.2,0.23,0.65,1.09,1.42]
#    Cmlexp = [0.104,0.008,-0.0965,-0.1975,-0.3055,-0.383]
#    Cl = np.zeros(len(alpha_range))
#    Cmle = np.zeros(len(alpha_range))
#    #Vortex Panel Method values for comparison
#    for a in alpha_range:    
#        Cl[i], Cmle[i] = VortexMethod(num_nodes,NACAnum,V_inf,a)
#        i+=1
#    #Plots and Tables
#    plt.figure(1)
#    plt.title('Lift Coefficient Comparison')
#    plt.scatter(alpha_range,Clexp, label = 'Experimental')
#    plt.scatter(alpha_range,Cl, label = 'Vortex Method Approximation')
#    plt.legend(loc = 'upper left')
#    plt.xlabel('Angle of Attack (degrees)')
#    plt.ylabel('Coefficient of Lift (C_l)')
#    table = [[0 for x in range(len(alpha_range))]for y in range(len(alpha_range))] 
#    for i in range(len(alpha_range)):
#        table[i] = (alpha_range[i],Cl[i],Clexp[i])
#    headers = ["Alpha","C_L Vortex Panel","C_L Experimental"]
#    t = tb.tabulate(table,headers,floatfmt=".6f")
#    print(t)           
#    plt.figure(2)
#    plt.title('Moment Coefficient Comparison')
#    plt.scatter(alpha_range,Cmlexp, label = 'Experimental')
#    plt.scatter(alpha_range,Cmle, label = 'Vortex Method Approximation')
#    plt.legend(loc = 'upper right')
#    plt.xlabel('Angle of Attack (degrees)')
#    plt.ylabel('Moment Coefficient (About Leading Edge)')
#    table = [[0 for x in range(len(alpha_range))]for y in range(len(alpha_range))] 
#    for i in range(len(alpha_range)):
#        table[i] = (alpha_range[i],Cmle[i],Cmlexp[i])
#    headers = ["Alpha","C_mLE Vortex Panel","C_mLE Experimental"]
#    t = tb.tabulate(table,headers,floatfmt=".6f")
#    print(t)       
#Plot_and_Tab(30,[0,0,12],1)           
            
            
            
            
