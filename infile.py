#-----------------------------------------------------------------------------#
#-------------------PARAMETERS FOR GROUND-STATE CALCULATION-------------------#
#-----------------------------------------------------------------------------#
c               =  5.0   # lattice constant
A               =  0.5   # potential parameter
B               =  0.35  # potential parameter

Nel             =  4     # number of electrons per unit cell

Kpoints         =  100   # number of k-points along x and y in the 1st quadrant of the BZ

Ksym            =  4     # 1: only first quadrant of BZ
                         # 2: only upper half of BZ
                         # 3: only right half of BZ
                         # 4: full BZ

Gnum            =  3     # G-vectors along x and y: -Gnum, -Gnum+1, ..., Gnum-1, Gnum
                         # The total number will be 2*Gnum+1 for each direction

Ntot            =  15    # total number of bands that we will include 
                         # (need more bands when calculating dielectric function)

HXC             =  0     # include Hartree and LDA XC? (0=no, 1=yes)

TOL             =  1.e-8 # numerical tolerance of self-consistency
MIX             =  0.8   # mixing parameter to accelerate self-consistency

GSout           =  3     # out=1: cross section of density
                         # out=2: 2D density
                         # out=3: band structure
                         # out=0: no output after ground-state calculation
#-----------------------------------------------------------------------------#
#-----------------PARAMETERS FOR CALCULATING EPSILON(OMEGA)-------------------#
#-----------------------------------------------------------------------------#
eps_omega_calc  =  1     # calculate epsilon(omega)? (0=no, 1=yes)

omegavalues     =  200   # number of frequency points 
d_omega         =  0.01  # spacing of omega-grid
eta             =  0.01  # imaginary part (line broadening parameter)

quasi2D         =  1     # 0: assume 2D Coulomb interaction 2 pi/k
                         # 1: assume 3D Coulomb interaction 4 pi/k**2
                     
kval            =  0.01  # small but finite k (only needed if quasi2D = 0)
#-----------------------------------------------------------------------------#
#------------------PARAMETERS FOR TIME-DEPENDENT CALCULATION------------------#
#-----------------------------------------------------------------------------#
timeprop        =  0     # do the time propagation? (0=no, 1=yes)

Tsteps          =  1000  # number of time steps
dt              =  0.5   # time step
Ncorr           =  0     # number of corrector steps 

Perturbation    =  3     # type of external perturbation:
                         # 0 = no perturbation
                         # 1 = pulsed scalar potential
                         # 2 = pulsed vector potential
                         # 3 = suddenly switched vector potential
                         # 4 = constant electric field
                  
alpha_xc        = -0.25  # LRC kernel parameter (should be <= 0)
#               = -0.25  # Notice: if we want a finite alpha_xc, we must set 
                         # the flag Current_mac = 1 (see below).

# if Perturbation=1, we assume that the external potential parameter A
# has a time dependence of the form A(t) = A*(1 + alpha*sin(omega_dr*t)*f(t)).
# Here, f(t) is a square pulse envelope of duration  Ncycles*2*pi/omega_dr

omega_dr        = 0.5   # frequency with which A is driven
alpha_t         = 0.1   # amplitude of the time-dependent potential perturbation
Ncycles         = 3     # number of cycles

# if Perturbation=2, we assume that we have a vector potential of strength E0
# along the angle theta (theta=0 along x-axis, theta=90 along y-axis etc.)
# with the same time-dependence as above (the case where Perturbation=1).

E0              = 0.5   # electric field amplitude
theta           = 0.0   # angle of the vector potential with respect to the x-axis

# if Perturbation=3, we assume that we have a vector potential of strength E0
# along the x-direction that is suddenly switched on at time t=0.

# if Perturbation=4, we assume there is a constant electric fiel of strength E0.

# The following output flags are 0=no, 1=yes:

Excited_pop     = 0     #  calculate the excited-state population at each t

Current_mac     = 0     #  calculate the macroscopic current density at each t

Occ_final       = 0     #  calculate the occupation numbers at the final time

Dielectric      = 0     #  calculate the time-dependent dielectric function. 
                        #  Note: this uses the same finite value of k ("kval") 
                        #  as the frequency-dependent dielectric function.
                 
Dipole          = 1     # calculate the time-dependent macroscopic dipole moment
