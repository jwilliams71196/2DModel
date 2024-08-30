import numpy  as     np
import cmath
from   numba  import njit
from   infile import c,Gnum,Kpoints,k_choice,B,D,alpha_xc,HXC

# Define the global prameters for these functions
N_G        = 2*Gnum+1
N_Gpts     = 2*N_G-1
dk         = (2 * np.pi) / (Kpoints * c)
G0         = (2 * np.pi) / c
q          = np.sqrt( k_choice[0]**2 + k_choice[1]**2 ) * dk
k          = np.zeros(Kpoints)
G          = np.zeros(N_G)
Gpts       = np.zeros(N_Gpts)
for ki in range (Kpoints): k[ki]    = (ki+0.5)*dk - np.pi/c
for Gi in range (N_G):     G[Gi]    = G0*(Gi-Gnum)
for Gi in range (N_Gpts):  Gpts[Gi] = G0*(Gi-2*Gnum)

#------------------------------------------------------------------------------
#    Umat_def defines the scalar potential
#------------------------------------------------------------------------------
def Umat_def(A,VHG,VXG):
    
    U    = np.zeros((N_G,N_G),       dtype=complex)
    Umat = np.zeros((N_G**2,N_G**2), dtype=complex)

    for Gxi in range (N_G):
        jx = Gxi-Gnum
        for Gyi in range (N_G):
            jy = Gyi - Gnum
            if ((jx == 0)      and (abs(jy) == 1)): U[Gxi,Gyi] = -(A-B)/2
            if ((jy == 0)      and (abs(jx) == 1)): U[Gxi,Gyi] = -(A-B)/2
            if ((abs(jx) == 1) and (abs(jy) == 1)): U[Gxi,Gyi] = -(A+B)/4
                
    for Gxi in range (N_G):
        jx = Gxi-Gnum
        for Gyi in range (N_G):
            jy = Gyi-Gnum
            if ((jy == 0) and (abs(jx) ==  1)): U[Gxi,Gyi] += -jx*1j*D/2

    U += VHG + VXG

    for Gxi in range (N_G):
        jx = Gxi-Gnum
        for Gyi in range (N_G):
            jy = Gyi-Gnum
            for Gxip in range (N_G):
                jxp = Gxip-Gnum
                for Gyip in range (N_G):
                    jyp = Gyip-Gnum
                
                    djx = jxp - jx
                    djy = jyp - jy
                
                    if ((abs(djx) <= Gnum) and (abs(djy) <= Gnum)):
                        Umat[Gxi+Gyi*N_G,Gxip+Gyip*N_G] = U[djx+Gnum,djy+Gnum]
    return Umat
#%%
@njit()
def Tmat_def(Tmat,kx,ky,G,Ax,Ay,Axcx,Axcy,Axc_count_x,Axc_count_y):
        
    for Gxi in range(N_G):
        for Gyi in range(N_G):
            Tmat[Gyi+Gxi*N_G,Gyi+Gxi*N_G] = \
                ((kx - G[Gxi] + Ax + Axcx + Axc_count_x[N_G+1,N_G+1])**2       \
               + (ky - G[Gyi] + Ay + Axcy + Axc_count_y[N_G+1,N_G+1])**2) / 2
                  
            for Gxip in range(N_G):
                for Gyip in range(N_G):
                   
                    Tmat[Gyi+Gxi*N_G,Gyi+Gxi*N_G] +=                           \
                        (2*kx - 2*G[Gxi] - G[Gxip]) / 2                        \
                        * Axc_count_x[Gxip-Gxi,Gyip-Gyi]                       \
                      + (2*ky - 2*G[Gyi] - G[Gyip]) / 2                        \
                        * Axc_count_x[Gxip-Gxi,Gyip-Gyi]
                    for Gxipp in range(N_G):
                        for Gyipp in range(N_G):
                            Tmat[Gyi+Gxi*N_G,Gyi+Gxi*N_G] +=                   \
                                Axc_count_x[Gxip-Gxipp, Gyip-Gyipp]            \
                              * Axc_count_x[Gxipp-Gxi,  Gyipp-Gyi] / 2
    
    return  Tmat
#%%
def VHartree(Gnum,nG):
    VHG2 = np.zeros((N_Gpts,N_Gpts), dtype=complex)
    VHG = np.zeros((N_G,N_G), dtype=complex)
    
    for Gxi in range(N_Gpts):
        for Gyi in range(N_Gpts):   
            if ((Gxi != N_G-1) and (Gyi != N_G-1)):  # except G=0
                VHG2[Gxi,Gyi] = (2 * np.pi) * nG[Gxi,Gyi]                      \
                                / np.sqrt(Gpts[Gxi]**2 + Gpts[Gyi]**2)

    for Gxi in range(N_G):
        for Gyi in range(N_G):
            VHG[Gxi,Gyi] = VHG2[Gxi+Gnum,Gyi+Gnum]
                
    return VHG     
#%%
def VLRC(nG,nGstat):

    VLRCG2 = np.zeros((N_Gpts,N_Gpts), dtype=complex)
    VLRCG = np.zeros((N_G,N_G), dtype=complex)
    
    for Gxi in range(N_Gpts):
        for Gyi in range(N_Gpts):   
            if ((Gxi != N_G-1) and (Gyi != N_G-1)):  # except G=0
                VLRCG2[Gxi,Gyi]= (2 * np.pi) * (nG[Gxi,Gyi] - nGstat[Gxi,Gyi]) \
                                 / np.sqrt(q**2 + Gpts[Gxi]**2 + Gpts[Gyi]**2)

    # This is analogous to the Hartree potential, but for the density response
    # We select an array of size N_G**2 which goes into the Hamiltonian
    for Gxi in range(N_G):
        for Gyi in range(N_G):
            VLRCG[Gxi,Gyi] = -(alpha_xc)/(4*np.pi) * VLRCG2[Gxi+Gnum,Gyi+Gnum]
                
    return VLRCG     
#%%
def Vexc(nG):

    Uvec = np.zeros (N_G**2,         dtype = complex)
    RHS  = np.zeros( N_G**2,         dtype = complex)
    Pmat = np.zeros((N_G**2,N_G**2), dtype = complex)
    den  = np.zeros((N_G,N_G),       dtype = float  ) 
    VXG  = np.zeros((N_G,N_G),       dtype = complex)

    # define the sampling points in real space  
    x  =  y = np.zeros(N_G)
    dx      = c / (N_G+1)
    
    for Gi in range(N_G): x[Gi] = y[Gi] = -c/2 + dx*(Gi+1)

    # calculate the real-space density at the sampling points
    for Gxi in range(N_G):
        for Gyi in range(N_G):  
            dum = 0
            for Gpxi in range(N_Gpts):
                for Gpyi in range(N_Gpts):
                    dum += nG[Gpxi,Gpyi]                                       \
                           * np.exp(1j*(Gpts[Gpxi]*x[Gxi] + Gpts[Gpyi]*y[Gyi]))
            den[Gxi,Gyi] = dum.real 

    # the right-hand side are the LDA-x potentials at the sampling points
    a0, a1, a2, a3 = -0.3568, 1.13, 0.9052, 0.4165

    for Gxi in range(N_G):
        for Gyi in range(N_G):
            
            rs =  1 / np.sqrt(np.pi * den[Gxi,Gyi])
            VX = -2 * np.sqrt(2) / (np.pi * rs)
            
            if   (HXC == 2):
                RHS[Gyi+Gxi*N_G] = VX
            elif (HXC == 3):                                 
                qq = np.sqrt(rs)                
                cc = 1 + a1 * qq + a2 * qq**2 + a3 * qq**3
                                
                VC = ( 1 +   2.00 * a1 * qq + (1.50 * a2 + a1**2) * qq**2      \
                         + ( 1.75 * a3      + (1.25 * a1 * a2))   * qq**3      \
                                            + ( 1.50 * a1 * a3)   * qq**4 )                              \
                     / ( cc**2 )
                
                VC = 0.5 * a0 * VC  
                
                RHS[Gyi+Gxi*N_G] = VX + VC

    # now define the coefficient matrix, we have:
    # V(x,y) = sum_{Gx,Gy} U_{Gx,Gy} * exp(-i * (Gx * x  Gy * y))
    for Gxi in range(N_G):
        for Gyi in range(N_G):
            G1 = Gyi + Gxi * N_G
            for Gpxi in range(N_G):
                for Gpyi in range(N_G):
                    G2 = Gpyi + Gpxi * N_G            
                    Pmat[G1,G2] = cmath.exp(-1j * G[Gxi] * x[Gpxi]             \
                                            -1j * G[Gyi] * y[Gpyi])         

    Uvec = np.linalg.solve(Pmat,RHS) 
    
    for Gxi in range(N_G):
        for Gyi in range(N_G):
            if ((Gxi != Gnum) and (Gyi != Gnum)): # exclude G=0 (constant shift)
                VXG[Gxi,Gyi] = Uvec[Gyi+Gxi*N_G]

    return VXG
