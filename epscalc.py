import numpy  as     np
from   scipy.fft     import fft
from   numba  import njit
from   infile import PrintOut,c,Ntot,Gnum,Kpoints,k_choice,om_pts,eta,d_omega, \
                     mode,quasi2D,alpha_xc,theta,dt,Tsteps,E0

# Define the 'universal' prameters for these functions
N_G        = 2*Gnum+1
N_Gpts     = 2*N_G-1
dk         = (2 * np.pi) / (Kpoints * c)
G0         = (2 * np.pi) / c
q          = np.sqrt( k_choice[0]**2 + k_choice[1]**2 ) * dk
qx, qy     = k_choice[0], k_choice[1]
k          = np.zeros(Kpoints)
G          = np.zeros(N_G)
Gpts       = np.zeros(N_Gpts)
theta      = theta * np.pi / 180
for ki in range (Kpoints): k[ki]    = (ki+0.5)*dk - np.pi/c
for Gi in range (N_G):     G[Gi]    = G0*(Gi-Gnum)
for Gi in range (N_Gpts):  Gpts[Gi] = G0*(Gi-2*Gnum)

#%%----------------------------------------------------------------------------
#    This subroutine calculates the time-dependent dipole moment
#------------------------------------------------------------------------------

@njit()
def dip_t_calc(OccNum,xi,ME):
   M = 0
   for kxi in range(Kpoints):
       for kyi in range(Kpoints):
           for l in range(Ntot):
               if (OccNum[l,kxi,kyi] == 1):
                   for n in range(Ntot):
                       for m in range(Ntot):
                           if (m != n):
                              M += xi[l,m,kxi,kyi] * np.conj(xi[l,n,kxi,kyi])  \
                                 * ME[n,m,kxi,kyi]        
   
   return (2 * M.imag * dk**2) / (2 * np.pi)**2 # factor 2 because of spin   

def dipole_t(OccNum,Ct,Cvec,ME,E):
    
    xi = np.zeros((Ntot,Ntot,Kpoints,Kpoints), dtype=complex)    
    
    for kxi in range (Kpoints):
        for kyi in range (Kpoints):
            for n in range (Ntot):
                if (OccNum[n,kxi,kyi] == 1):
                    for m in range(Ntot):
                        DIP = 0
                        for Gi in range (N_G**2):    
                            DIP +=         Ct  [n,kxi,kyi,Gi]                  \
                                 * np.conj(Cvec[m,kxi,kyi,Gi])
                        xi[n,m,kxi,kyi] = DIP

    return dip_t_calc(OccNum,xi,ME)

#%%----------------------------------------------------------------------------
# This subroutine calculates the dipole matrix elements ME (along the x-axis).
# Notice that we want to avoid matrix elements between near-degenerate
# bands. To do this we demand that the energies are separated by more than 0.1,
# which happens here (*). The fudge factor 0.1 will need to be explained later.
#------------------------------------------------------------------------------

@njit()
def ME_calc_fun(Cvec,E,n,m,kxi,kyi):
    
    M = 0
    
    for Gxi in range (N_G):
        for Gyi in range (N_G):
            Gi = Gyi + Gxi*N_G
            M += np.conj(Cvec[n,kxi,kyi,Gi]) * Cvec[m,kxi,kyi,Gi]              \
               * (G[Gxi] * np.cos(theta) + G[Gyi] * np.sin(theta))
    return M / (E[n,kxi,kyi] - E[m,kxi,kyi]) 

def ME_calc(OccNum,Cvec,E):
    
    ME = np.zeros((Ntot,Ntot,Kpoints,Kpoints), dtype = complex) 

    for kxi in range (Kpoints):
        for kyi in range (Kpoints):
            for n in range (Ntot):
                for m in range (Ntot):
                    if ((OccNum[m,kxi,kyi] != OccNum[n,kxi,kyi])               \
                    or  (abs(m-n) > 1)                                         \
                    or  (abs(E[n,kxi,kyi]-E[m,kxi,kyi]) > 0.1)):
                        ME[n,m,kxi,kyi] = ME_calc_fun(Cvec,E,n,m,kxi,kyi)
    return ME

#%%---------------------------------------------------------------------------#
# This subroutine calculates the LRC frequency-dependent dielectric function  #
# by solving the Dyson equation, in the head-only approximation               #
#-----------------------------------------------------------------------------#

@njit()
def chi0_calc(OccNum,ME,E,om):
    chi = 0
    for kxi in range(Kpoints):
        for kyi in range(Kpoints):
            for n in range(Ntot):
                if (OccNum[n,kxi,kyi] == 1):
                    for l in range(Ntot):
                        if (OccNum[l,kxi,kyi] == 0):
                            chi += abs(ME[n,l,kxi,kyi])**2                     \
                              * (1/(E[n,kxi,kyi] - E[l,kxi,kyi] + om + 1j*eta) \
                              +  1/(E[n,kxi,kyi] - E[l,kxi,kyi] - om - 1j*eta))
    return chi * (2 * c**2 * dk**2) / (2 * np.pi)**2 # factor 2 because of spin

def epsilon_om(OccNum,ME, E,FX):
    
    omega   = np.zeros(om_pts, dtype = float  )
    epsilon = np.zeros(om_pts, dtype = complex)    
    
    fx_zero = FX[4*Gnum, 4*Gnum]  # this is fxc_{GG'} for G = G' = 0
        
    omPathPercent = om_pts // 10 + 1
    percentDone   = 0
                      
    for wi in range (om_pts):
        om = wi*d_omega
        omega[wi] = om

        if ((PrintOut == 1) and ((wi%omPathPercent) == 0)):
            percentDone += 10
            print(str(percentDone) + "%-", end = '')
        
        # first, calculate the noninteracting response function 
        # chi0_00(k-->0,omega)
        chi = chi0_calc(OccNum,ME,E,om)
            
        # RPA
        if   (mode == 1): chi_proper = chi
        # ALDA head-only
        elif (mode == 2): chi_proper = chi / (1 - fx_zero * chi / c**4)
        # LRC head-only
        elif (mode == 3) and (quasi2D == 0): chi_proper =                      \
                     chi / (1 - (alpha_xc / (4 * np.pi)) * 2 * np.pi * chi * q)
        elif (mode == 3) and (quasi2D == 1): chi_proper =                      \
                     chi / (1 - (alpha_xc / (4 * np.pi)) * 4 * np.pi * chi)
        
        if   (quasi2D == 0): epsilon[wi] = 1 - (2 * np.pi * q) * chi_proper
        elif (quasi2D == 1): epsilon[wi] = 1 - (4 * np.pi)     * chi_proper
        
    if (PrintOut == 1): print("100%]\n")
        
    return epsilon, omega  

#%%---------------------------------------------------------------------------#
# these are the matrix elements for the finite-k calculation                  #
#-----------------------------------------------------------------------------#

@njit()
def MEk_calc_fun1(Cvec,n,kxi,kyi,l,i3,k3):
    fun1 = 0
    for Gxi in range(N_G):
        for Gyi in range(N_G):
            Gi    = Gyi + Gxi * N_G
            fun1 += np.conj(Cvec[n,kxi,kyi,Gi]) * Cvec[l,i3,k3,Gi] 
     
    return fun1

@njit()
def MEk_calc_fun2(Cvec,n,kxi,kyi,l,i3,k3p):
    fun2 = 0
    for Gxi in range(N_G):
        for Gyi in range(1,N_G):
            ind1  =  Gyi      + Gxi * N_G
            ind2  = (Gyi - 1) + Gxi * N_G
            fun2 += np.conj(Cvec[n,kxi,kyi,ind1]) * Cvec[l,i3,k3p,ind2] 
     
    return fun2

@njit()
def MEk_calc_fun3(Cvec,n,kxi,kyi,l,i3p,k3):
    fun3 = 0
    for Gxi in range(1, N_G):
        for Gyi in range(N_G):
            ind1 = Gyi +  Gxi      * N_G
            ind2 = Gyi + (Gxi - 1) * N_G
            fun3 += np.conj(Cvec[n,kxi,kyi,ind1]) * Cvec[l,i3p,k3,ind2] 
     
    return fun3

@njit()
def MEk_calc_fun4(Cvec,n,kxi,kyi,l,i3p,k3p):
    fun4 = 0
    for Gxi in range(1, N_G):
        for Gyi in range(1, N_G):
            ind1 =  Gyi      +  Gxi      * N_G
            ind2 = (Gyi - 1) + (Gxi - 1) * N_G
            fun4 += np.conj(Cvec[n,kxi,kyi,ind1]) * Cvec[l,i3p,k3p,ind2]  
     
    return fun4                              

def MEk_calc(k_choice,OccNum,Cvec):  

    MEk = np.zeros((Ntot,Ntot,Kpoints,Kpoints),dtype=complex)  
    
    for n in range (Ntot):
        for l in range (Ntot):

            for kxi in range (Kpoints):
                for kyi in range (Kpoints):
            
                    i3  = kxi + qx
                    k3  = kyi + qy
                    i3p = i3  - Kpoints
                    k3p = k3  - Kpoints
                    
                    ME = 0
                    
                    if (i3 < Kpoints) and (k3 < Kpoints):                               
                       if (OccNum[n,kxi,kyi] != OccNum[l,i3,k3]):                                                                                                      
                           ME += MEk_calc_fun1(Cvec,n,kxi,kyi,l,i3,k3)
 
                    if (i3 < Kpoints) and (k3 >= Kpoints):
                        if (OccNum[n,kxi,kyi] != OccNum[l,i3,k3p]):
                            ME += MEk_calc_fun2(Cvec,n,kxi,kyi,l,i3,k3p)
                            
                    if (i3 >= Kpoints) and (k3 < Kpoints):
                        if (OccNum[n,kxi,kyi] != OccNum[l,i3p,k3]):
                            ME += MEk_calc_fun3(Cvec,n,kxi,kyi,l,i3p,k3)
                
                    if (i3 >= Kpoints) and (k3 >= Kpoints):                        
                        if (OccNum[n,kxi,kyi] != OccNum[l,i3p,k3p]):
                            ME += MEk_calc_fun4(Cvec,n,kxi,kyi,l,i3p,k3)
                            
                    MEk[n,l,kxi,kyi] = ME

    return MEk

#%%---------------------------------------------------------------------------#
# omega- and k-dependent dielectric function                                  #
#-----------------------------------------------------------------------------#

@njit()
def chi0k_calc(OccNum,MEk,E,om):
    
    chi = 0
    
    for kxi in range(Kpoints):
        for kyi in range(Kpoints):
            
            i3 = kxi + k_choice[0]*dk
            k3 = kyi + k_choice[1]*dk
            
            if (i3 >= Kpoints): i3 = i3-Kpoints
            if (k3 >= Kpoints): k3 = k3-Kpoints               
            
            for n in range(Ntot):
                if (OccNum[n,kxi,kyi] == 1):
                    for l in range(Ntot):
                        if (OccNum[l,i3,k3] == 0):
                            chi += ( abs(MEk[n,l,kxi,kyi])**2                  \
                                   / (E[n,kxi,kyi] - E[l,i3,k3] + om + 1j*eta) \
                                 -   abs(MEk[l,n,kxi,kyi])**2                  \
                                   / (E[l,kxi,kyi] - E[n,i3,k3] + om + 1j*eta))
    
    return chi * (2 * c**2 * dk**2) / (2 * np.pi)**2 # factor 2 because of spin

def epsilon_k_om(OccNum,MEk,E,FX):
    
    omega   = np.zeros(om_pts, dtype = float  )
    epsilon = np.zeros(om_pts, dtype = complex)
      
    fx_zero = FX[4*Gnum,4*Gnum]  # this is fxc_{GG'} for G = G' = 0
    
    print("q =", q)
     
    for wi in range(om_pts):
        omega[wi] = wi*d_omega
        if ((wi%10) == 0): print(wi, "out of", om_pts)
        
        # first calculate the noninteracting response function chi0_00(k,omega)        
        chi = chi0k_calc(OccNum,MEk,E,omega[wi])
            
        # RPA
        if    (mode == 1): chi_proper = chi
        # ALDA head-only
        elif  (mode == 2): chi_proper = chi / (1 - fx_zero * chi / c**4)
        # LRC head-only
        elif ((mode == 3) and (quasi2D == 0)): chi_proper = chi                \
                    / (1 - (alpha_xc / (4 * np.pi)) * (2 * np.pi / q)    * chi)
        elif ((mode == 3) and (quasi2D == 1)): chi_proper = chi                \
                    / (1 - (alpha_xc / (4 * np.pi)) * (4 * np.pi / q**2) * chi)
        
        # Calculate the dielectric function from the response function
        if   (quasi2D == 0): epsilon[wi] = 1 - (2 * np.pi / q)    * chi_proper
        elif (quasi2D == 1): epsilon[wi] = 1 - (4 * np.pi / q**2) * chi_proper
        
    return epsilon, omega

def eps_FFT_d(Dipole_t,Time):

    damp = np.zeros(Tsteps,  dtype = float)
    for ti in range(Tsteps): damp[ti] = np.exp(-Time[ti] * 0.01)

    domega = (2 * np.pi) / (Tsteps * dt)
    OMM    = np.zeros(Tsteps,dtype = float)
    for ti in range(Tsteps): OMM[ti] = ti * domega 
    
    return (1 - fft(Dipole_t * damp) * ( 2 * np.pi * q * c**2 * dt) / E0), OMM
