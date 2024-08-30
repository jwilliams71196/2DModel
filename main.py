import time
start_time = time.time()
#-----------------------------------------------------------------------------#
# This program calculates a 2D band structure for the "egg carton" potential. #
# and then goes on to do the time propagation after a pulse.                  #
#-----------------------------------------------------------------------------#
import matplotlib.pyplot as plt
import numpy             as np
import scipy
import sys
from   scipy.fft     import fft
# from scipy.interpolate import CubicSpline as cspline

# Import functions
from mat_def     import Umat_def,Tmat_def,Vexc,VHartree,VLRC
from dencalc     import densityG,pop_calc,current,A_xc_LRC_con_mac,n_sq_calc
from Band_output import den_diag_out,den_full_out,band_out
from epscalc     import epsilon_om,dipole_t,ME_calc,MEk_calc,epsilon_k_om,     \
                        eps_FFT_d
from Dyson       import fxc_calc,dy_calc

# Import parameters
from infile      import c,A,Nel,Kpoints,Gnum,Ntot,Tsteps,dt,HXC,               \
                        TOL,MIX,out,restart,Ncorr,PrintOut,                    \
                        eps_omega_calc,k_choice,om_pts,d_omega,eta,quasi2D,    \
                        mode,alpha_xc,beta_xc,gamma_xc,                        \
                        timeprop,Proca,TDLRC,Perturbation,                     \
                        omega_dr,alpha_t,Ncycles,E0,theta,                     \
                        Excited_pop,Current_mac,Dipole,CounterTerm
# Covert theta to radians
theta    = theta * np.pi / 180 # convert to radians
# Define the reciprocal space grid parameters
N_G    = (2 * Gnum)  + 1
N_Gpts = (2 * N_G)   - 1
dk     = (2 * np.pi) / (Kpoints * c)
G0     = (2 * np.pi) / c
q      = np.sqrt( k_choice[0]**2 + k_choice[1]**2 ) * dk
# Define the k- and G-vectors
k    = np.zeros(Kpoints)
G    = np.zeros(N_G)
Gpts = np.zeros(N_Gpts)
for ki in range (Kpoints): k   [ki] = dk * (ki + 0.5) - (np.pi / c)
for Gi in range (N_G):     G   [Gi] = G0 * (Gi -     Gnum)
for Gi in range (N_Gpts):  Gpts[Gi] = G0 * (Gi - 2 * Gnum)
# Initialize arrays for the Hamiltonian and its arts
Hmat                = np.zeros((N_G**2,N_G**2),               dtype = complex)
Tmat                = np.zeros((N_G**2,N_G**2),               dtype = complex)
Umat                = np.zeros((N_G**2,N_G**2),               dtype = complex)
# Initialize arrays for the energy and occupation numbers
Eaux                = np.zeros( Ntot*Kpoints**2,              dtype =  float )
E                   = np.zeros((Ntot,Kpoints,Kpoints),        dtype =  float )
OccNum              = np.zeros((Ntot,Kpoints,Kpoints),        dtype =   int  )
ME                  = np.zeros((Ntot,Ntot,Kpoints,Kpoints),   dtype = complex)
# Initialize arrays for the density
nG                  = np.zeros((N_Gpts,N_Gpts),               dtype =  float )
nG0                 = np.zeros((N_Gpts,N_Gpts),               dtype =  float )
nG1                 = np.zeros((N_Gpts,N_Gpts),               dtype =  float )
nGstat              = np.zeros((N_Gpts,N_Gpts),               dtype =  float )
# Initialize arrays for the Hartree and XC potentials
VHG                 = np.zeros((N_G,N_G),                     dtype = complex)
VHG_old             = np.zeros((N_G,N_G),                     dtype = complex)
VXG                 = np.zeros((N_G,N_G),                     dtype = complex)
VXGstat             = np.zeros((N_G,N_G),                     dtype = complex)
VXGdyn              = np.zeros((N_G,N_G),                     dtype = complex)
VXG_old             = np.zeros((N_G,N_G),                     dtype = complex)
VLRCG               = np.zeros((N_G,N_G),                     dtype = complex)
# Initialize the arrays for the eigenvectors
Cvec                = np.zeros((Ntot,Kpoints,Kpoints,N_G**2), dtype = complex)
C0                  = np.zeros((Ntot,Kpoints,Kpoints,N_G**2), dtype = complex)
Ct                  = np.zeros((Ntot,Kpoints,Kpoints,N_G**2), dtype = complex)
Ct1                 = np.zeros((Ntot,Kpoints,Kpoints,N_G**2), dtype = complex)
Cvect               = np.zeros((Ntot,Kpoints,Kpoints,N_G**2), dtype = complex)
# Initialize arrays for the time-propogation algorithm
ONE                 = np.zeros((N_G**2,N_G**2),               dtype =  float )
PHI                 = np.zeros((N_G**2),                      dtype = complex)
RHS                 = np.zeros((N_G**2),                      dtype = complex)
Mmat                = np.zeros((N_G**2,N_G**2),               dtype = complex)
Mmat0               = np.zeros((N_G**2,N_G**2),               dtype = complex)
# Initialize arrays in which to store the time-dependent data
Time,      Nex      = np.zeros( Tsteps ), np.zeros( Tsteps )
Epsilon_t, Dipole_t = np.zeros( Tsteps ), np.zeros( Tsteps )
jx_mac,    jy_mac   = np.zeros( Tsteps ), np.zeros( Tsteps )
jx_para,   jy_para  = np.zeros( Tsteps ), np.zeros( Tsteps )
Axc_x,     Axc_y    = np.zeros( Tsteps ), np.zeros( Tsteps )
A_xc_con_t_x        = np.zeros((Tsteps,   N_Gpts,   N_Gpts))
A_xc_con_t_y        = np.zeros((Tsteps,   N_Gpts,   N_Gpts))
# XC Coefficients
if   (Proca == 0):
    c_J  = alpha_xc * dt**2
    c_n1 = 2
    c_n2 = 1
# 'Proca' Equation Coefficients,  Central Difference Method
elif (Proca == 1):
    c_J  = (    alpha_xc * dt**2 ) / (1 + beta_xc * dt / 2)
    c_n1 = (2 - gamma_xc * dt**2 ) / (1 + beta_xc * dt / 2)
    c_n2 = (1 - beta_xc * dt / 2 ) / (1 + beta_xc * dt / 2)
# 'Proca' Equation Coefficients, Backward Difference Method
elif (Proca == 2):
    c_J  = (alpha_xc * dt**2) / (1 + beta_xc * dt + gamma_xc * dt**2)
    c_n1 = (2 + beta_xc * dt) / (1 + beta_xc * dt + gamma_xc * dt**2)
    c_n2 = (        1       ) / (1 + beta_xc * dt + gamma_xc * dt**2)
# 
F_occ = np.zeros( Kpoints**2 * Ntot) 
Et    = np.zeros((Ntot,Kpoints,Kpoints)) 
# Initialize the identity matrix
for i in range(N_G**2): ONE[i,i] = 1
# Initialize the time-dependent calculation variables
T,At,Ax,Ay,Axcx,Axcy = 0,A, 0, 0, 0, 0
Axc_con_x,Axc_con_y  = np.zeros((N_Gpts,N_Gpts)), np.zeros((N_Gpts,N_Gpts))

'''---------------------------------------------------------------------------#
#                           ______              ____                          #
#                         /  ____ \           /  __ \                         #
#                        / /     \ \         / /   \ \                        #
#                       | |      |_|        | |    |_|                        #
#                       | |   _____         \ \____                           #
#                       | |  |  __ |         \_____ \                         #
#                       | |  |_| / /         _    | |                         #
#                        \ \____/ /        | |___/ /                          #
#                         \______/ ROUND   \______/ TATE                      #
#                                                                             #
#-----------------------------------------------------------------------------#
#                     START OF SELF-CONSISTENCY LOOP                          #
#                  (solve for each k-point independently)                     #
#---------------------------------------------------------------------------'''

# Set up the iteration variables for the SC loop
Eref          = 1
iteration_num = 0

# Read in GS data from a file, if desired
if   (restart == 0):
    pass
elif (restart == 1):
    with open("save.txt", "") as f:
        Gi = -1
        for line in f.readlines():
            Gi += 1
            Gxi, Gyi = Gi//N_G, Gi%N_G
            #list1 = line.split()
            #VHG[Gxi,Gyi], VXG[Gxi,Gyi] = complex(list1[0]), complex(list1[1])
            VHG[Gxi,Gyi], VXG[Gxi,Gyi] = complex(line.split())
else:
    print("Invalid value for 'restart', must be 0 or 1.")
    sys.exit()

# Solve the GS-KSE self-consistently
while (abs(Eref - E[1,0,0]) > TOL):
    iteration_num += 1   
    
    # Set an upper bound of itterations so the code doesn't go infinitely
    if iteration_num > 1000: break
    
    # Print the progress to the screen, if desired
    if (PrintOut == 1):
        print("Iteration no.",       iteration_num,                            \
              "\tdisagreement: = ",  np.round(abs(Eref - E[1,0,0]), 6)) 
    
    # we want to reference the previous itteration at the end
    Eref = E[1,0,0]

    # Calculate the GS Potential Energy matrix
    Umat = Umat_def(A,VHG,VXG)

    # We solve the equation for each k-point seperately
    for kxi in range (Kpoints):
        for kyi in range (Kpoints): 
            
            # Add all of the G-vector contributions
            for Gxi in range(N_G):
                for Gyi in range(N_G):
                    Tmat[Gyi + Gxi * N_G, Gyi + Gxi * N_G] =                   \
                        0.5*( (k[kxi] - G[Gxi])**2 + (k[kyi] - G[Gyi])**2 )
            
            # Solve the GS-KES for its eigenvaules and eigenvectors
            Hmat = Tmat + Umat
            vals, vecs = np.linalg.eigh(Hmat)
            for n in range(Ntot):
                E[n,kxi,kyi],Eaux[n+kyi*Ntot+kxi*Kpoints*Ntot]=vals[n],vals[n]
                for Gi in range (N_G**2):
                    Cvec[n,kxi,kyi,Gi] = vecs[Gi,n]

    # Find the Fermi energy and the occupation numbers, print if desired
    sorted   = np.argsort(Eaux)  
    EF_index = (Kpoints**2 * Nel) // 2 - 1
    EF       = Eaux[sorted[EF_index]]
    EF1      = Eaux[sorted[EF_index+1]]
    if (PrintOut == 1):
        print(  "EF:",    np.round(EF,    6),                                  \
              "\tEF+1:",  np.round(EF1,   6),                                  \
              "\tE_gap:", np.round(EF1-EF,6))
    
    # Identify the initialy occupied and unnoccupied states
    for kxi in range(Kpoints):
        for kyi in range(Kpoints):
            for n in range(Ntot):
                if (E[n,kxi,kyi] <= EF): OccNum[n,kxi,kyi] = 1
                else:                    OccNum[n,kxi,kyi] = 0

    # Calculate the GS density
    nG = densityG(Cvec,OccNum)
    
    # Update the Hartree and Exchange terms
    if   (HXC == 0): # in this case we want only 1 iteration
        Eref    = E[1,0,0]
    elif (HXC <= 3): # otherwise we need to mix the old and new
        VHG_old = VHG.copy()           
        VHG     = VHartree(Gnum,nG,Gpts)   
        VHG     = MIX * VHG + (1 - MIX) * VHG_old
    
        if (HXC > 1):
            VXG_old = VXG.copy() 
            VXG     = Vexc(HXC,Gnum,nG,G,Gpts,c)
            VXG     = MIX * VXG + (1 - MIX) * VXG_old
    else:
        print("Invalid value for 'HXC'. Must be 0, 1, 2, or 3.")
        sys.exit()

#------------------------------------------------------------------------------
#                      END OF SELF-CONSISTENCY LOOP
#------------------------------------------------------------------------------
if   (out == 0): pass
elif (out == 1): den_diag_out(nG)
elif (out == 2): den_full_out(nG)
elif (out == 3): band_out(Umat,EF)
else:
    print("Invalid value for 'out', must be 0, 1, 2, or 3.")
    sys.exit()
    
with open("save.txt", "w") as f:
    for Gxi in range(N_G):
        for Gyi in range(N_G):
            f.write(str(VHG[Gxi,Gyi])+ "\t" + str(VXG[Gxi,Gyi]) + "\n")

GStime = time.time()

if (PrintOut == 1): print("\n--- ground state completed in %s seconds ---\n"   \
                          % np.round(GStime-start_time, 6))

if (Dipole == 1) or (eps_omega_calc == 1):
    ME = ME_calc(OccNum,Cvec,E)

'''---------------------------------------------------------------------------#
#                     _                _____                                  #
#                    | |              |  __ \                                 #
#                    | |              | |  \ \                                #
#                    | |              | |   | |                               #
#                    | |              | |__/ /                                #
#                    | |              |  __ \                                 #
#                    | |              | |  \ \                                #
#                    | |______        | |   \ \                               #
#                    |________| INEAR |_|    \_\ ESPONSE                      #
#-----------------------------------------------------------------------------#
#   Calculate the frequency-dependent dielectric function:                    #
#---------------------------------------------------------------------------''' 

if (eps_omega_calc > 0):
    FXC = fxc_calc(Gpts,nG)
    if (PrintOut == 1):
        print("q =", np.round(q, 6))
        print("\ncalculating the dielectric function for", om_pts, "points: [",\
               end='')       

    if (eps_omega_calc == 1):                  
        if (mode < 10):
            epsilon,omega = epsilon_om(OccNum,ME,E,FXC)
        if (mode > 10):
            epsilon,omega = dy_calc(k_choice,FXC,om_pts,d_omega,eta,           \
                                    OccNum,Ntot,Kpoints,Kpoints,Gnum,Gpts,     \
                                    Cvec,dk,ME,E,c,quasi2D,mode,alpha_xc)
                
    if (eps_omega_calc == 2):  
        MEk = MEk_calc(k_choice,OccNum,Ntot,Kpoints,Kpoints,N_G,Cvec)
        
        epsilon,omega = epsilon_k_om(k_choice,om_pts,d_omega,eta,OccNum,       \
                                     Ntot,Kpoints,Kpoints,dk,MEk,E,c,quasi2D,  \
                                     mode,alpha_xc,FXC,Gnum)    
        
    plt.plot(omega,epsilon.real)
    plt.plot(omega,epsilon.imag)
    plt.title ("Optical Specrum from Linear Response" +                        \
               "\n $\\alpha_{xc}$ = " + str(alpha_xc))
    plt.xlabel("$\\omega$")
    plt.ylabel("$\\epsilon$($\\omega$)")
    plt.xlim([0,om_pts*d_omega])
    plt.show()
        
    with open("eps_omega_alpha=" + str(alpha_xc) + ".txt", "w") as f:
        for wi in range(om_pts):
            f.write(str(np.round(omega[wi],5)) + "  "                          \
                  + str(np.round(epsilon.real[wi],5)) + "\t"                   \
                  + str(np.round(epsilon.imag[wi],5)) + "\n")
    
    LRtime = time.time()
    calc_time = LRtime - GStime
    
    if (PrintOut == 1):
        print("epsilon_0 =", np.round(epsilon[0], 6))
        
        days      = calc_time // (24 * 3600)
        calc_time = calc_time  % (24 * 3600)
        hours     = calc_time // 3600
        calc_time = calc_time  % 3600
        minutes   = calc_time // 60
        seconds   = np.floor(calc_time % 60)
        
        print("\n--- Linear response calculation completed in " +              \
              "%d days:%02d hours:%02d minutes:%02d seconds ---\n"             \
              %(days, hours, minutes, seconds))

else: LRtime = GStime
        
#------------------------------------------------------------------------------
#                      END OF LINEAR RESPONSE CALCULATION
#------------------------------------------------------------------------------

if   (timeprop == 0)     : sys.exit() # stop here if we only want gnp.round state
elif (1 <= timeprop <= 3): pass
else:
    print("Invalid value for 'timeprop', must be 0, 1, 2, or 3")
    sys.exit()

'''---------------------------------------------------------------------------#
#                          _______     _______________                        #
#                         |  ____ \   |______   ______|                       #
#                         | |    \ \         | |                              #
#                         | |    | |         | |                              #
#                         | |____/ /         | |                              #
#                         |  ___  /          | |                              #
#                         | |   \ \          | |                              #
#                         | |    \ \         | |                              #
#                         |_|     \_\ EAL    |_| IME                          #
#                                                                             #
#-----------------------------------------------------------------------------#
#                                                                             #
# the following portion of code is the time-propagation which calculates the  #
# excited state population, macroscopic current density, and/or the dipole    #
# momment, as indicated in the input file.                                    #
#                                                                             #
#-----------------------------------------------------------------------------#
#                           BEGIN TIME PROPAGATION                            #
#---------------------------------------------------------------------------'''

# INITIALIZATIONS
Ct, nGstat, n_sq = np.copy(Cvec), np.copy(nG), n_sq_calc(nG)

for ti in range(Tsteps): 
    T        += dt
    Time[ti]  =  T
    T_half    =  T - (dt/2) # evaluate perturbation at mid-timestep
    
    # Print to the screen every 10 time-steps, if desired
    if ((((ti+1)%10) == 0) and (PrintOut == 1)): 
        print("\ntime   =", np.round(T, 6)) 

    # sin^2 Envelope of the time-dependent perturbation
    if ((Perturbation == 1) or (Perturbation == 2)):
        Tpulse = Ncycles*2*np.pi/omega_dr
        ENV = 0    
        if (T_half < Tpulse): ENV = np.sin(np.pi * (T_half / Tpulse))**2    

    # scalar potential:
    if (Perturbation == 1): 
        At = A * (1 + (alpha_t * np.sin(omega_dr * T_half) * ENV)) 

    # vector potential x and y components:
    if (Perturbation == 2):
        Ax = ((E0/omega_dr) * np.sin(omega_dr*T_half) * ENV) * np.cos(theta)
        Ay = ((E0/omega_dr) * np.sin(omega_dr*T_half) * ENV) * np.sin(theta) 

    if (Perturbation == 3): Ax, Ay = E0*np.cos(theta), E0*np.sin(theta)

#------------------------- PREDICTOR-CORRECTOR LOOP --------------------------#

    pc_count = -1

    while (pc_count < Ncorr):
        pc_count += 1

        # Predictor Step
        if (pc_count == 0):
            
            # Include the LRC, if desired
            if (TDLRC > 0):
                if (Proca < 2):
                    Axc_x[ti]  = c_J  * jx_mac[ti-1] * (dt**2*c**2/2)          \
                               + c_n1 * Axc_x [ti-1]                           \
                               - c_n2 * Axc_x [ti-2]
                    Axc_y[ti]  = c_J  * jy_mac[ti-1] * (dt**2*c**2/2)          \
                               + c_n1 * Axc_y [ti-1]                           \
                               - c_n2 * Axc_y [ti-2]
                else:
                    Axc_x[ti]  = c_J  * jx_mac[ ti ] * (dt**2*c**2/2)          \
                               + c_n1 * Axc_x [ti-1]                           \
                               - c_n2 * Axc_x [ti-2]
                    Axc_y[ti]  = c_J  * jy_mac[ ti ] * (dt**2*c**2/2)          \
                               + c_n1 * Axc_y [ti-1]                           \
                               - c_n2 * Axc_y [ti-2]
                
                Axcx, Axcy = Axc_x[ti], Axc_y[ti]
            
            # Include the counter-term, if desired
            if (CounterTerm > 0) and (TDLRC != 0):
                A_xc_con_t_x[ti] = A_xc_LRC_con_mac(A_xc_con_t_x,jx_mac,       \
                                                    nGstat,n_sq,ti)
                A_xc_con_t_y[ti] = A_xc_LRC_con_mac(A_xc_con_t_y,jy_mac,       \
                                                    nGstat,n_sq,ti)
                
                Axc_con_x,Axc_con_y = A_xc_con_t_x[ti],A_xc_con_t_y[ti]

            # Recalculate the density, if needed
            if (HXC > 0) or (TDLRC == 2):
                nG0 = nG = densityG(Kpoints,Kpoints,dk,Ct,OccNum) 
        
        if (HXC > 0): VHG = VHartree(Gnum,nG,Gpts)    
        if (HXC > 1): VXG = Vexc(HXC,Gnum,nG,G,Gpts,c)

        # Alter the XC terms, if needed
        if (TDLRC == 2):
            VLRCG = VLRC(Gnum,nG,nGstat,alpha_xc,Gpts,q)
            VXG  += VLRCG
        
        # Define the time-dependent potential energy matrix
        Umat = Umat_def(At,VHG,VXG)
        
        # Create a copy of the potential energy matrix, if needed
        if (pc_count == 0) and (timeprop == 3): Umat0 = Umat.copy()

        # Solve the TD-KSE individually for each k-point
        for kxi in range (Kpoints):
            for kyi in range (Kpoints):  

                # Define the kinetic energy terms
                Tmat = Tmat_def(Tmat, k[kxi], k[kyi], G, Ax, Ay,               \
                                Axcx, Axcy, Axc_con_x, Axc_con_y)
                
                # Construct the Hamiltonian    
                Hmat = Tmat + Umat
                
                # Construct a copy of the Hamiltonian, if needed
                if (timeprop == 3): Hmat0 = Tmat + Umat0
                
                # Define the matracies for the time-propogation
                for n in range(Ntot):
                    if OccNum[n,kxi,kyi]==1:
                        for Gi in range(N_G**2):
                            PHI[Gi] = Ct[n,kxi,kyi,Gi]
                        if   (timeprop == 1):    
                            Mmat    = ONE - dt*1j/2 * Hmat            
                            RHS     = Mmat.dot(PHI)
                            Mmat    = ONE + dt*1j/2 * Hmat
                            PHI     = scipy.linalg.solve(Mmat, RHS,            \
                                                         assume_a = "gen")
                        elif (timeprop == 2):
                            Mmat    = -dt*1j*Hmat 
                            ExpMAT  = scipy.linalg.expm(Mmat)
                            PHI     = np.matmul(ExpMAT,PHI)  
                        elif (timeprop == 3):
                            Mmat    = -dt*1j/2 * Hmat0
                            Mmat2   = np.matmul(Mmat,Mmat)                                                        
                            Mmat3   = np.matmul(Mmat,Mmat2)
                            Mmat4   = np.matmul(Mmat,Mmat3) 
                            ExpMAT0 = ONE + Mmat + Mmat2/2 + Mmat3/6 + Mmat4/24
                            RHS     = np.matmul(ExpMAT0,PHI)                                                        
                            
                            Mmat    = -0.5*dt*1j*Hmat
                            Mmat2   = np.matmul(Mmat,Mmat)
                            Mmat3   = np.matmul(Mmat,Mmat2)
                            Mmat4   = np.matmul(Mmat,Mmat3)  
                            ExpMAT  = ONE + Mmat + Mmat2/2 + Mmat3/6 + Mmat4/24
                            PHI     = np.matmul(ExpMAT,RHS)
                        else:
                            print("Invalid value for input paramete",          \
                                  "'timeprop'."                                \
                                  "Must be 0, 1, 2, or 3.")
                        
                        # Initilize the Slater orbital wave functions
                        for Gi in range(N_G**2):
                            Ct1[n,kxi,kyi,Gi] = PHI[Gi]   

        # this is the corrector step
        if (Ncorr > 0):
            
            # Calculate the LRC components & macroscopic current density
            if ((TDLRC > 0) and (ti > 0)):
                
                jx, jy     = current(OccNum,Ct1,G)        
                jx_mac[ti] = jx-(Ax+Axc_x[ti]+Axc_con_x[N_G+1,N_G+1])*Nel/c**2    
                jy_mac[ti] = jy-(Ay+Axc_y[ti]+Axc_con_y[N_G+1,N_G+1])*Nel/c**2
                    
                if (Proca < 2):
                    Axc_x[ti]  = c_J  * jx_mac[ti-1] * (dt**2*c**2/2)          \
                               + c_n1 * Axc_x [ti-1]                           \
                               - c_n2 * Axc_x [ti-2]
                    Axc_y[ti]  = c_J  * jy_mac[ti-1] * (dt**2*c**2/2)          \
                               + c_n1 * Axc_y [ti-1]                           \
                               - c_n2 * Axc_y [ti-2]
                else:
                    Axc_x[ti]  = c_J  * jx_mac[ ti ] * (dt**2*c**2/2)          \
                               + c_n1 * Axc_x [ti-1]                           \
                               - c_n2 * Axc_x [ti-2]
                    Axc_y[ti]  = c_J  * jy_mac[ ti ] * (dt**2*c**2/2)          \
                               + c_n1 * Axc_y [ti-1]                           \
                               - c_n2 * Axc_y [ti-2]
                
                Axcx, Axcy = Axc_x[ti], Axc_y[ti]

                # Counter Term
                if (CounterTerm > 0):
                    A_xc_con_t_x[ti] =                                         \
                        A_xc_LRC_con_mac(A_xc_con_t_x,jx_mac,nGstat,n_sq,ti)
                    A_xc_con_t_y[ti] =                                         \
                        A_xc_LRC_con_mac(A_xc_con_t_y,jy_mac,nGstat,n_sq,ti)

            # Calculate an auxilary denisty, if needed
            if ((HXC>0) or (TDLRC==2)): nG1 = densityG(Ct1,OccNum)  
            
            # Correct the XC and related terms
            if ((timeprop == 1) or (timeprop == 2)): 
                Axcx      = 0.5 * (Axc_x[ti-1]        + Axc_x[ti])
                Axcy      = 0.5 * (Axc_y[ti-1]        + Axc_y[ti])
                Axc_con_x = 0.5 * (A_xc_con_t_x[ti-1] + A_xc_con_t_x[ti])
                Axc_con_y = 0.5 * (A_xc_con_t_y[ti-1] + A_xc_con_t_y[ti])
                nG        = 0.5 * (nG0                + nG1)
                
            if (timeprop == 3): 
                Axcx,      Axcy      = Axc_x[ti], Axc_y[ti]
                Axc_con_x, Axc_con_y = A_xc_con_t_x[ti], A_xc_con_t_y[ti]
                nG                   = nG1.copy()

    # End of predictor-corrector loop. Ready for the next time step.                          
    Ct = Ct1.copy()

#-----------------------------------------------------------------------------#
# calculate the excited-state population                                      #
#-----------------------------------------------------------------------------#
    if (Excited_pop == 1):
        N_gs = pop_calc(OccNum,Cvec,Ct)
        N_ex = Nel-N_gs
        
        Nex[ti] = N_ex
        
        if (((ti+1)%10) == 0) and (PrintOut == 1): 
            print("N_ex   =", np.round(N_ex, 6))  
#-----------------------------------------------------------------------------#
# calculate the macroscopic current density                                   #
#-----------------------------------------------------------------------------#
    if (Current_mac == 1):
        jx, jy = current(OccNum,Ct,G)
        
        # Paramagnetic current
        jx_para[ti], jy_para[ti] = jx, jy
        
        # Total current
        jx_mac[ti]   = jx - (Nel/c**2)                                         \
                       * ( Ax + Axc_x[ti] + A_xc_con_t_x[ti,N_G+1,N_G+1] )
        jy_mac[ti]   = jy - (Nel/c**2)                                         \
                       * ( Ay + Axc_y[ti] + A_xc_con_t_y[ti,N_G+1,N_G+1] )

        if (((ti+1)%10) == 0) and (PrintOut == 1):
            print(  "jx_mac =",np.round(jx_mac[ti], 6),                        \
                  "\tjy_mac =",np.round(jy_mac[ti], 6))

#-----------------------------------------------------------------------------#
# calculate the time-dependent dipole moment                                  #
#-----------------------------------------------------------------------------#
    if (Dipole == 1):  
        DIP = dipole_t(OccNum,Ct,Cvec,ME,E) 
        
        Dipole_t[ti] = DIP
        
        if (((ti+1)%10) == 0) and (PrintOut == 1):
            print("Dip    =", np.round(DIP, 6))
#-----------------------------------------------------------------------------#
#  check to make sure the calculation hasn't 'blown up'                       #
#-----------------------------------------------------------------------------#
    if  np.isnan(Dipole_t[ti]) or np.isnan(Nex[ti])                            \
    or  np.isnan(jx_mac[ti])   or np.isnan(jy_mac[ti]):                        
        Dipole_t[ti] = 0.0
        jx_mac[ti],   jy_mac[ti] = 0.0, 0.0
        Nex[ti] = 0.0
        print("Calculation blew up at time step", (ti+1))
        break
#-----------------------------------------------------------------------------#
RTtime = time.time()
calc_time = RTtime - LRtime
if (PrintOut == 1):
    days      = calc_time // (24 * 3600)
    calc_time = calc_time  % (24 * 3600)
    hours     = calc_time // 3600
    calc_time = calc_time  % 3600
    minutes   = calc_time // 60
    seconds   = np.floor(calc_time % 60)
    
    print("\n--- Real time calculation completed in " +                        \
          "%d days:%02d hours:%02d minutes:%02d seconds ---\n"                 \
          %(days, hours, minutes, seconds))

'''---------------------------------------------------------------------------#
#                           END OF TIME PROPAGATION.                          #
#-----------------------------------------------------------------------------#
#      ______    _      _   ___________   _____     _      _   ___________    #
#     /  __  \  | |    | | |____   ____| |  __ \   | |    | | |____   ____|   #
#    |  /  \  | | |    | |      | |      | |  \ \  | |    | |      | |        #
#    | |    | | | |    | |      | |      | |   | | | |    | |      | |        #
#    | |    | | | |    | |      | |      | |__/ /  | |    | |      | |        #
#    | |    | | | |    | |      | |      |  ___/   | |    | |      | |        #
#    |  \__/  | |  \__/  |      | |      | |       |  \__/  |      | |        #
#     \______/   \______/       |_|      |_|        \______/       |_|        #
#                                                                             #
#---------------------------------------------------------------------------'''

print("Writing and plotting outputs...")

if (Excited_pop == 1):
    
    # Write the excited state population to a file
    
    with open("Nex_alpha=" + str(alpha_xc) + ".txt", "w") as f:
        for ti in range(Tsteps):
            f.write(str(np.round(Time[ti],5))      + "\t" +                    \
                    str(np.round(Nex[ti].real,12)) + "\n" )
    
    # Plot the excited state population
    title = "Time-Dependent Excited State Population"                          \
          + "\n with $\\alpha_{xc}$ = "       + str(alpha_xc)             
    if (Proca > 0):
        title += "\t with $\\beta_{xc}$ = "   + str(beta_xc)                   \
              +  "\t with $\\gamma_{xc}$ = "  + str(gamma_xc)
    plt.title (title)
    plt.xlabel("time")
    plt.ylabel("excited state population\n(as a percentage)")
    plt.plot(Time,Nex)
    plt.show()

if (Current_mac == 1):
    
    # Write the time-dependent macroscopic current
    # density and vector potential each to a file
    
    with open("j_alpha=" + str(alpha_xc) + ".txt", "w") as f:
        for ti in range(Tsteps):
            f.write(str(np.round(Time  [ti],      5)) + "\t" +                 \
                    str(np.round(jx_mac[ti].real,12)) + "\t" +                 \
                    str(np.round(jy_mac[ti].real,12)) + "\n")
    
    with open("A_alpha=" + str(alpha_xc) + ".txt", "w") as f:
        for ti in range(Tsteps):
            f.write(str(np.round(Time [ti],      5))  + "\t" +                 \
                    str(np.round(Axc_x[ti].real,12))  + "\t" +                 \
                    str(np.round(Axc_y[ti].real,12)))
        
    f.close()    
    
    # Plot the macroscopic current density
    title = "Time-Dependent Macroscopic Current Density"                       \
          + "\n with $\\alpha_{xc}$ = "       + str(alpha_xc)             
    if (Proca > 0):
        title += "\t with $\\beta_{xc}$  = "  + str(beta_xc)                   \
              +  "\t with $\\gamma_{xc}$ = "  + str(gamma_xc)
    plt.title (title)
    plt.xlabel("time")
    plt.ylabel("macroscopic current density")
    plt.plot(Time, jx_mac, label = "x")
    plt.plot(Time, jy_mac, label = "y")

    plt.legend()
    plt.show()

if (Dipole == 1):
    
    # Write the time-dependent dipole moment to a file
    
    with open("dip_t_alpha=" + str(alpha_xc) + ".txt", "w") as f:
        for ti in range(Tsteps):
            f.write(str(np.round(Time[ti],6)) + "  " \
                  + str(np.round(Dipole_t[ti],12)) +  '\n')
    
    # Plot the time-dependent dipole moment
    title = "Time-Dependent Dipole Momment"                                    \
          + "\n with $\\alpha_{xc}$ = "       + str(alpha_xc)             
    if (Proca > 0):
        title += "\t with $\\beta_{xc}$  = "  + str(beta_xc)                   \
              +  "\t with $\\gamma_{xc}$ = "  + str(gamma_xc)
    plt.title (title)
    plt.xlabel("time")
    plt.ylabel("dipole momment")
    plt.plot(Time,Dipole_t)
    plt.show()
            
    # Fourier Transform (FFT) the dipole moment to obtain the
    # time-dependent dielectric function
    '''
    
    eps_t, OMM = eps_FFT_d(Dipole_t,Time)
    
    '''
    tf = Tsteps * dt

    Time      = np.zeros(Tsteps, dtype = float  )
    damp      = np.zeros(Tsteps, dtype = float  )
    integrand = np.zeros(Tsteps, dtype = complex)

    for ti in range(Tsteps):
        T        = ti * dt
        Time[ti] = T
        damp[ti] = np.exp(-T * 0.01)

    domega = (2 * np.pi) / tf
    OMM    = np.zeros(Tsteps)

    for ti in range(Tsteps):
        OMM[ti] = ti * domega 

    eps_t = fft(Dipole_t * damp) * dt / E0

    eps_t = 1 - eps_t * ( 2 * np.pi * q * c**2 )
    
    
    # Plot the TD and LR spectra
    if (eps_omega_calc > 0):
        eps_max = max(max(eps_t.real),   max(-eps_t.imag),                     \
                      max(epsilon.real), max(epsilon.imag)) + 3
        eps_min = min(min(eps_t.real),   min(-eps_t.imag),                     \
                      min(epsilon.real), min(epsilon.imag)) - 3
            
        title = "Comparison of Optical Spectra using the LRC"                  \
              + "\n Between Linear Response and Real-Time Methods"             \
              + "\n with $\\alpha_{xc}$ = "                                    \
              + str(alpha_xc)
        if (Proca > 0):
            title += "\t with $\\beta_{xc}$ = "    + str(beta_xc)              \
                  +  "\t with $\\gamma_{xc}$ = "   + str(gamma_xc)
    else:
        eps_max = max(max(eps_t.real), max(-eps_t.imag)) + 3
        eps_min = min(min(eps_t.real), min(-eps_t.imag)) - 3
        title = "Optical Spectra using Real-Time LRC"                          \
              + "\n with $\\alpha_{xc}$ = "  + str(alpha_xc)             
        if (Proca > 0):
            title += "\t with $\\beta_{xc}$ = "   + str(beta_xc)               \
                  +  "\t with $\\gamma_{xc}$ = "  + str(gamma_xc)

    plt.title (title)
    plt.xlabel('$\omega$')
    plt.ylabel('$\epsilon$($\omega$)')
    plt.xlim(0.0,2.0)
    plt.ylim(eps_min,eps_max)
    
    plt.plot(OMM,  eps_t.real,                                                 \
             label = "RT real", linestyle = "-",                               \
             color = "red",     linewidth = 0.9)
    plt.plot(OMM, -eps_t.imag,                                                 \
             label = "RT imag", linestyle = "-",                               \
             color = "blue",    linewidth = 0.9)
        
    if (eps_omega_calc > 0):

        plt.plot(omega, epsilon.real,                                          \
                 label = "LR real", linestyle = "--",                          \
                 color = "red",     linewidth = 0.9)
        plt.plot(omega, epsilon.imag,                                          \
                 label = "LR imag", linestyle = "--",                          \
                 color = "blue",    linewidth = 0.9)

    plt.legend()
    plt.show()

end_time  = time.time()
run_time  = end_time - start_time

days      = run_time // (24 * 3600)
run_time  = run_time  % (24 * 3600)
hours     = run_time // 3600
run_time  = run_time  % 3600
minutes   = run_time // 60
seconds   = np.floor(run_time % 60)

print("\n--- Run completed in %d days:%02d hours:%02d minutes:%02d seconds ---"\
      %(days, hours, minutes, seconds))
