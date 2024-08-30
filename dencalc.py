import numpy  as     np
from   numba  import njit
from   infile import c,Nel,Ntot,Gnum,alpha_xc,dt,CounterTerm,Kpoints,k_choice

# Define the 'universal' prameters for these functions
N_G        = (2 * Gnum) + 1
N_Gpts     = (2 * N_G)  - 1
dk         = (2 * np.pi) / (Kpoints * c)
G0         = (2 * np.pi) / c
q          = np.sqrt( k_choice[0]**2 + k_choice[1]**2 ) * dk
k          = np.zeros(Kpoints)
G          = np.zeros(N_G)
Gpts       = np.zeros(N_Gpts)
for ki in range (Kpoints): k[ki]    = (ki+0.5)*dk - np.pi/c
for Gi in range (N_G):     G[Gi]    = G0*(Gi-Gnum)
for Gi in range (N_Gpts):  Gpts[Gi] = G0*(Gi-2*Gnum)

#%%---------------------------------------------------------------------------#
# This subroutine calculates the density in G-space nG
# using occupation numbers and simple integration
#-----------------------------------------------------------------------------#
@njit()
def dc(Cvec,OccNum,nGk,nG):

    for kxi in range(Kpoints):
        for kyi in range(Kpoints):
            for n in range(Ntot):
                if OccNum[n,kxi,kyi]==1:
                    for Gx1 in range(N_G):
                        for Gy1 in range(N_G): 
                            G1 = Gy1 + Gx1*N_G 
                            C1 = Cvec[n,kxi,kyi,G1]
                            for Gx2 in range(N_G):
                                Gx = Gx2-Gx1+2*Gnum                                   
                                for Gy2 in range(N_G):
                                    G2 = Gy2 + Gx2*N_G 
                                    C2 = Cvec[n,kxi,kyi,G2]                                                                  
                                    Gy = Gy2-Gy1+2*Gnum
                                                                                
                                    nGk[Gx,Gy,kxi,kyi] += C1*np.conj(C2)
    
    for Gxi in range(N_Gpts):
        for Gyi in range(N_Gpts):
            density = 0
            for kxi in range(Kpoints):
                for kyi in range(Kpoints):   
                    density += nGk[Gxi,Gyi,kxi,kyi]
    
            nG[Gxi,Gyi] = (2 * density * dk**2) / (2 * np.pi)**2

    return nG

def densityG(Cvec,OccNum):
    
    nG  = np.zeros((N_Gpts,N_Gpts),                 dtype=complex)
    nGk = np.zeros((N_Gpts,N_Gpts,Kpoints,Kpoints), dtype=complex)        
                    
    return dc(Cvec,OccNum,nGk,nG)

#%%---------------------------------------------------------------------------#
#   This subroutine calculates the macroscopic paramagnetic current density   #
#-----------------------------------------------------------------------------#

@njit() 
def current_calc(OccNum,Ct,G):

    jx, jy = 0, 0
    
    for kxi in range(Kpoints):
        for kyi in range(Kpoints):            
            for n in range (Ntot):   
                if OccNum[n,kxi,kyi]==1:
                    for Gxi in range(N_G):
                        for Gyi in range(N_G):
                            m = Gyi + Gxi*N_G
                            jx += 2*dk**2/(2*np.pi)**2 \
                               *  G[Gxi]*np.abs(Ct[n,kxi,kyi,m])**2
                            jy += 2*dk**2/(2*np.pi)**2 \
                               *  G[Gyi]*np.abs(Ct[n,kxi,kyi,m])**2
    
    return jx, jy

def current(OccNum,Ct,G):
    
    jx, jy = current_calc(OccNum,Ct,G)
    
    return jx.real, jy.real

#%%---------------------------------------------------------------------------#
# This function calculates the absolutesquare of the density                  #
#-----------------------------------------------------------------------------#
def n_sq_calc(nG): # G-dependent density (array[N_Gpts,N_Gpts])

    n_sq = 0

    for Gxi in range(N_Gpts):
        for Gyi in range(N_Gpts):
            n_sq += c**2 * (nG[Gxi,Gyi] * nG[N_Gpts-1-Gxi,N_Gpts-1-Gyi]).real

    return n_sq

#%%---------------------------------------------------------------------------#
# This function calculates the LRC zero-force theorem 'counter' vec potential #
#-----------------------------------------------------------------------------#
def A_xc_LRC_con_mac(A_xc_counter_t,j,nG,n_sq,ti):

    A_xc_counter = np.zeros((N_Gpts,N_Gpts))
    
    # head only
    if   (CounterTerm == 1):
        A_xc_counter[N_G+1,N_G+1] = - ( alpha_xc * dt * Nel**2 / c**2 )        \
                                    * ( nG[N_G+1,N_G+1] / n_sq )               \
                                    * ( 2 * j[ti-1] - j[ti-2] )                \
                                  + 2 * A_xc_counter_t[ti-1,N_G+1,N_G+1]       \
                                  -     A_xc_counter_t[ti-2,N_G+1,N_G+1]
    
    # G-vector Dependent
    elif (CounterTerm == 2):
        for Gxi in range(N_Gpts):
            for Gyi in range(N_Gpts):
                A_xc_counter[Gxi,Gyi] = - ( alpha_xc * dt * Nel**2 / c**2 )    \
                                        * ( nG[Gxi,Gyi] / n_sq )               \
                                        * ( 2 * j[ti-1] - j[ti-2])             \
                                        + ( 2 * A_xc_counter_t[ti-1,Gxi,Gyi] ) \
                                        -       A_xc_counter_t[ti-2,Gxi,Gyi]
    
    return A_xc_counter

#%%---------------------------------------------------------------------------#
# This function calculates the number of electrons remaining in ground state  #
#-----------------------------------------------------------------------------#
@njit() 
def pop_calc(OccNum,Ct,Cvec):

    pop = 0   
    for kxi in range (Kpoints):
        for kyi in range (Kpoints):
            # Itterate over only pairs of occupied bands
            for n in range (Ntot):
                if (OccNum[n,kxi,kyi] == 1):
                    for m in range(Ntot):
                        if (OccNum[m,kxi,kyi] == 1):                          
                            popG = 0
                            for k in range (N_G**2):                                 
                                popG +=         Ct  [n,kxi,kyi,k]              \
                                      * np.conj(Cvec[m,kxi,kyi,k])                                                                                              
                            pop += np.abs(popG)**2
        
    return (2 * dk**2 * c**2) / (2 * np.pi)**2 * pop # factor 2 because of spin
