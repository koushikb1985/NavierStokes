#navierStokes.py
#
#Solves 2D force-free Navier-Stokes equation assuming perriodic boundary conditions.
#We use the streamfunction-vorticity formalism and the equations are solved using spectral 
#methods. In this method, the equations in Fouruer domain is solved:
#
# \del/\del_t \omegahat_p  = -FFT(u . \nabla \omega)_p - \nu p^2 \omegahat_p
# -p^2 \psihat_p = \omegahat_p
# u_i = \epsilon_{ij} \del_j \psi 
# \omega = \epsilon_{ij} \del_j u_i
# \omegahat_p = Fourier Transform(omega) with Fourier momentum p
#
# Inverse Fourier Transform is used to obtain u, \omega and \psi in real space
#
#This code will display a "movie" at the end of the computation
#
import os
from numpy import *
from fastnumpy import * # This is required for MKL FFT.
# This can be installed using enthought's package manager
# NumPy 1.6 is reuired to install fastnumpy
from fastnumpy.mklfft import *
from time import *
from matplotlib import *
from pylab import *
from mpl_toolkits.mplot3d import Axes3D # Plotting package
import matplotlib.animation as animation # Required for making animations
from numexpr import * # This is required for fast array operations
# This can be installed using enthought's package manager

        
def viscosity(k2, omegahat): 
    # This function computes Laplacian in Fourier space
    # This is called viscosity as it was intially used just for computing viscosity. 
    # The author of this code did not change the name.
    # Takes k2 = (i*p)^2  and the fourier transform of omega (omegahat) as input
    # evaluate is a function in numexpr that computes the espression within coutes
    # this function utilizes all available threads by default
    # kx, ky are already multiplied by 1j, that is k2 = -p^2

        
        return evaluate("k2*omegahat")
                
def Poisson(invk2, omegahat): 
    # This function computes psi or solve Poisson's equation in  Fourier Space
    # It takes invK2 = 1/k2 = - 1/p^2 and omegahat as input and returns -omegahat/p^2 which is Fourier transform 
    # of psi. It is essential to remove the zero mode in p^2 before this step.
    # kx, ky are already multiplied by 1j, that is k2 = -p^2 or invk2 = -/p^2
        
        return evaluate("omegahat*invk2") # \psihat_p = \omegahat_p/p^2 = invk2*omegahat

def convective(kx,ky,psihat,omegahat):
    # This function computes FFT(u . \nabla \omega)_p 
    # kx, ky are already multiplied by 1j
    # fft2 is 2D fast fourier transform and ifft2 is the inverse of fft2 
    # fft2 and ifft2 make use of mkl library.
        ux  = ifft2( evaluate("ky*psihat")).real
        uy  = ifft2(evaluate("-kx*psihat")).real
        domega_dx =ifft2(evaluate("kx*omegahat")).real
        domega_dy =ifft2(evaluate("ky*omegahat")).real
        conv = evaluate("ux*domega_dx + uy*domega_dy")
        return fft2(conv) # returns the fourier transform of convective term

class FluidDynamics:

        """Class that describes two-dimensional incompressible flow"""
        
        def __init__(self,nx=512,ny=512, nu2=0.0, dt =.0005): 
            # This is a constructor that intializes all the arrays required for the computation.
            # Note that the fourier modes kx and ky are alos initialized here
            # nx and ny are the number of grid points along x and y
            # nu2 is the coefficient of second order viscosity
            # dt is duration of a single time step
            # Values for nx,ny, nu2 and dt are taken to be 512, 512, 0.0 and .0005 if they are not explicitly specified
        
                self.dt = dt # defines a memeber function that takes the value of dt
                self.nx=nx # defines a memeber function that takes the value of nx
                self.ny=ny # defines a memeber function that takes the value of ny
                self.nu2 = nu2 # defines a memeber function that takes the value of nu2
                self.psi0 = zeros((self.nx, self.ny)) # psi0 is the streamfunction in real space
                self.omega0 = zeros((self.nx, self.ny)) # omega0 is the vorticity in real space
                self.x = 2.0*pi/nx*arange(nx) # Fourier Grid along x
                self.y = 2.0*pi/ny*arange(ny) # Fourier Grid along y
                """ Define Fourier Modes """
                # Note that kx and ky are two-dimensional arrays
                # mat command converts arrays to matrices 
                # For matrices * operator denotes matrix multiplication (Python feature)
                # Note that definitions for kx, ky take care of the  Nyquist criterion
                # kx  = i px
                # ky = i py
                self.kx=array(1j*mat(ones((1,ny))).T*mat(mod(arange(1,nx+1)-ceil(nx/2+1),nx)-nx/2)) 
                self.ky=array(1j*mat(mod(arange(1,ny+1)-ceil(ny/2+1),ny)-ny/2).T*mat(ones((1,nx))))
                """ Compute square of the Fourier Modes """
                # k2 = - p^2 =  px^2 + py^2
                # This is required for computing second derivative of fields
                self.k2 = self.kx*self.kx + self.ky*self.ky
                """ Compute inverse of the square of the Fourier Modes """
                # This is required for solving Poisson Equation
                # kp2 is same as k2 with zero mode removed
                # invkp2 = -1/p^2 = 1/kp2 (with zero mode removed)
                # It is essential to specify or remove the zero mode while solving Poisson equation
                # with periodic boundary conditions. 
                self.kp2 = copy(self.k2)
                self.kp2[0,0] = 1.0 # Removes Zero Mode 
                # Note that the right hand side can be set to any non-zero value, this does not affect velocity
                # but changes psi_original to psi_original + f(t) where f(t) is some function of t.
                # This change does not affect velocity
                self.invKp2 = 1.0/self.kp2 # inverse of p^2 with zero mode removed
               
        def initialState(self): 
            # This is a memberfunction that sets initial condition for psi, omega , omegahat, psihat, etc

                U0 = 1.0 # Amplitude of velocity
                wavelength = 1/10.0 #sets periodicity of inital velocity profile
                epsilon = 0.3 #amplitude of fluctuations
                psi = zeros((self.nx, self.ny)) # initialize the psi array to zero
                psif =  zeros((self.nx, self.ny)) # initialize the psi array to zero
                xx,yy = meshgrid(self.x,self.y) # defines a 2D mesh 
                for m in xrange(1,4): # Fluctuation is written as a superposition of 4 by 4 fourier modes 
                        for n in xrange(1,4):
                                c_xy = 2.0*random()-1.0
                                psif += c_xy*cos(m*xx)*sin(n*yy) 
                psi =U0*cos(xx/wavelength) + epsilon*psif # compute psi of counter flow + flutuations
                self.psi0 = psi # assign the value of psi to the member function psi0
                self.psi0hat = fft2(self.psi0) # compute psihat from psi
                self.omega0hat = viscosity(self.k2, self.psi0hat) # compute omegahat from psihat
                # Note that viscoity function just computes the Laplacian
                self.omega0 = ifft2(self.omega0hat).real
                
                
        def velocity(self): 
            # This is a member function that computes/returns velocity in real space
            #kx, ky are already multiplied by 1j
                self.ux  = ifft2( self.ky*self.psi0hat).real
                self.uy  = ifft2(-self.kx*self.psi0hat).real

        def omega(self):
            # This is a member function that computes/returns \omega in real space
            
                self.omega  = ifft2(self.omega0hat).real
              
        def domega_dt(self, omegahat, psihat):

                # This is a member function that computes the source term describing the time evolution of vorticity
                # It returns domega_dt
                return ((-convective(self.kx,self.ky,psihat,omegahat)\
                                          + self.nu2 * viscosity(self.k2,omegahat)))
                
        def integrateRK(self, t=1.0): 
            # This function implements Fourth order Runge-Kutta integration (explicit)
            # It does the integration for t/dt time steps
            # t is taken to be 1.0 if not specified
 
                steps = int(t/self.dt)
                dtloc = self.dt # defines a local variable which is same as self.dt
                omegaloc = self.omega0hat # defines a local variable which is same as self.omegahat
                for i in xrange(steps):
                        k1 = self.domega_dt(self.omega0hat,self.psi0hat) 
                        psi1hat = Poisson(self.invKp2, evaluate("omegaloc+0.5*dtloc*k1"))
                        k2 = self.domega_dt(evaluate("omegaloc+0.5*dtloc*k1"),psi1hat)
                        psi2hat = Poisson(self.invKp2, evaluate("omegaloc+0.5*dtloc*k2"))
                        k3 = self.domega_dt(evaluate("omegaloc+0.5*dtloc*k2"),psi2hat)
                        psi3hat = Poisson(self.invKp2, evaluate("omegaloc+dtloc*k3"))
                        k4 = self.domega_dt(evaluate("omegaloc+dtloc*k3"),psi3hat)
                        self.omega0hat +=evaluate("dtloc/6.0*(k1+2.0*k2+2.0*k3+k4)")
                        self.psi0hat = Poisson(self.invKp2,self.omega0hat)
 

time_initial = time()
print "\nInitializing......"
numx = 512 # Sets number of gridpoints along x
numy = 512 # Sets number of gridpoints along y
nuval = 1.0/300.0 # Sets value of viscosity
fluid = FluidDynamics(nx=numx,ny=numy, nu2=nuval, dt = .0005) # Create an instance of an object with nx, ny, nu2, dt specified
fluid.initialState() # Defines initial state of psi
xx,yy=meshgrid(fluid.x,fluid.y) # Defines a 2D mesh for plotting
steps = 400 # Number of big steps or number of frames in the final animation
trate = 10.0 # Number of small steps.  big time step = trate*small step = trate*dt

time_final = time()
tinitial = time_final - time_initial
print "\nTime to initialize is ", tinitial, "s"
print "\nGrid Size = ",numx, " by ", numy
print "\ndt =", fluid.dt  

tcurr=0.0
tprev =0.0
os.path.expanduser("~/KBTurbulence.txt")
with file('KBTurbulence.txt','w') as plotfile: # opens KBTurbulence.txt in write mode
    for i in xrange(steps): #Big Steps
        tprev = tcurr
        tcurr +=fluid.dt*trate
        time_initial = time()
        fluid.integrateRK(t=tcurr-tprev) #Integrate for trate*dt units of time
        psi = ifft2(fluid.psi0hat).real # Compute psi in real space
        omega = ifft2(fluid.omega0hat).real # compute omega in real space
        time_final = time()
        tcomp = time_final - time_initial
        print "\nPhysical Time =", tcurr,",   Max Omega=", omega.max(), "    computation time (for ", int(trate), " time steps) is ", tcomp, "s",";    computation time (for 1 time step) is", tcomp/trate, "s"
        time_initial = time()
        plotfile.write('Physical Time: {0}\n'.format(tcurr))
        savetxt(plotfile, omega, fmt='%-7.2f')
        time_final = time()
        twrite = time_final - time_initial
        print "\nTime to write into a file =", twrite