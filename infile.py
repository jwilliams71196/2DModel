#-----------------------------------------------------------------------------#
#---------------PARAMETERS FOR CALCULATING EPSILON(OMEGA)---------------------#
#-----------------------------------------------------------------------------#
PrintOut      =      1  # print outputs to the screen? still will be written
                        # to a file.
                        # 1 = yes
                        # 0 = no

#-----------------------------------------------------------------------------#
#-----------------PARAMETERS FOR GROUND-STATE CALCULATION---------------------#
#-----------------------------------------------------------------------------#
c             =    5.0  # lattice constant

# Direct   Gap Example: A = 2.50, B = 2.30, Nel = 8
# Indirect Gap Example: A = 0.8,  B = 0.7,  Nel = 4
 
A             =   0.80  # cosine potential parameter
B             =   0.70  # cosine potential parameter
D             =   0.00  # sinusoidal potential parameter (if desired)

Nel           =      4  # number of electrons per unit cell

Kpoints       =     20  # number of k-points along x and y             

Gnum          =      4  # G-vectors along x and y: -Gnum, -Gnum+1, ..., Gnum
                        # The total number will be 2*Gnum+1 for each direction

Ntot          =     10  # total number of bands that we will include 
                        # (may need more when calculating dielectric function)

HXC           =      0  # 0: noninteracting
                        # 1: Hartree only
                        # 2: Hartree + LDA exchange
                        # 3: Hartree + LDA exchange-correlation

TOL           = 1.e-12  # numerical tolerance of self-consistency
MIX           =    0.8  # mixing parameter to accelerate self-consistency

out           =      0  # out=1: cross section of density
                        # out=2: 2D density
                        # out=3: band structure
                        # out=0: no output after ground-state calculation
               
restart       =      0  # 0: after convergence, save VH and VXC to a file
                        # 1: read VH and VXC from file. 

#-----------------------------------------------------------------------------#
#---------------PARAMETERS FOR CALCULATING EPSILON(OMEGA)---------------------#
#-----------------------------------------------------------------------------#
eps_omega_calc =     1  # calculate epsilon(omega)?  
                        # 0 = no 
                        # 1 = yes, for k=0
                        # 2 = yes, for finite k

k_choice       =  (1,0) # k_choice = [qx,qy]; qx and qy are (0,1,...,Kpoints).
                        # The k-vector for which we calculate epsilon(k,omega)
                        # has the format Kx = qx*dk and Ky = qy*dk, where
                        # dk is the k-grid spacing (dk = PI*Kpoints/c).

om_pts         =   150  # number of frequency points 
d_omega        =  0.01  # spacing of omega-grid
eta            =  0.01  # imaginary part (line broadening parameter)

quasi2D        =     0  # 0: assume 2D Coulomb interaction 2 pi/k
                        # 1: assume 3D Coulomb interaction 4 pi/k**2

mode           =     3  #  1: RPA
                        #  2: LDA-XC (head-only approximation)
                        #  3: LRC (head-only approximation)
                        # 12: LDA-XC, full Dyson equation 
                        # 13: LRC,    full Dyson equation
                        # 14: LRC',   headless Dyson equation
                        # NOTE: Dyson is only implemented for the k=0

alpha_xc       = -0.00  # LRC kernel parameter (should be <= 0)

#-----------------------------------------------------------------------------#
#---------------PARAMETERS FOR TIME-DEPENDENT CALCULATION---------------------#
#-----------------------------------------------------------------------------#
timeprop       =     2  # Time propagation mode:
                        # 0: no time propagation
                        # 1: Crank-Nicolson algorithm
                        # 2: exponential midpoint rule (recomended)
                        # 3: ETRS 4th order (not recommended)

Tsteps         =  2000  # number of time steps
dt             =  0.25  # time step
Ncorr          =     1  # number of corrector steps (must be >= 0)

TDLRC          =     1  # use the time-dependent LRC potential? 
                        # 0 = no
                        # 1 = yes, head-only
                        # 2 = yes, including local-field effects
                        # If yes, one must set the flag Current_mac = 1

CounterTerm    =     0  # include the counter term?
                        # 0 = no
                        # 1 = yes, head only
                        # 2 = yes, G-dependent
                        # 3 = yes, TD-density

Proca          =     1  # include the 'Proca' terms?
                        # 0 = no
                        # 1 = yes, Central  Difference Method
                        # 2 = yes, Backward Difference Method
                        
beta_xc        =  0.00  # 'Proca'/Damping kernel parameter (should be >= 0)
gamma_xc       =  0.00  # 'Proca'/Damping kernel parameter (should be >= 0)

Perturbation   =     3  # type of external perturbation:
                        # 0 = no perturbation
                        # 1 = pulsed scalar potential A(t) 
                        # 2 = pulsed vector potential
                        # 3 = suddenly switched vector potential 

# if Perturbation = 1, we assume that the external potential parameter A
# has a time dependence of the form A(t) = A*(1 + alpha*sin(omega_dr*t)*f(t)).
# Here, f(t) is a sine-square pulse envelope of duration  Ncycles*2*pi/omega_dr
#
# if Perturbation=2 or 3, we assume that we have a vector potential of strength
# E0 along the angle theta (theta=0 along x-axis, theta=90 along y-axis etc.)
#
# Perturbation=2: the vector potential has the same time dependence 
# (sine-square pulse) as the scalar potential in Perturbation=1).
# 
# Perturbation=3: the vector potential is suddenly switched on at time t=0.
# This corresponds to a delta-peaked electric field pulse.

omega_dr       =   0.5  # frequency with which A is driven
alpha_t        = 0.010  # amplitude of time-dependent potential perturbation
Ncycles        =     3  # number of cycles
                        # (omega_dr,alpha_t,Ncycles not used if Perturbation=3)

E0             = 0.010  # electric field amplitude
theta          =    45  # angle of the vector potential wrt the x-axis

# The following output flags are 0 = no, 1 = yes:

Excited_pop    =     1  # calc & plot the excited-state population at each t
Current_mac    =     1  # calc & plot the macro current density at each t
Dipole         =     1  # calc & plot the time-dependent macro dipole moment

#*****************************************************************************#
#***************************USER INPUT ENDS HERE******************************#
#*****************************************************************************#