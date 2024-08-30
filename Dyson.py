import cmath
import numpy  as     np
from   numba  import njit
from   infile import c,Gnum,Kpoints,k_choice,HXC

# Define the 'universal' prameters for these functions
N_G        = 2 * Gnum   + 1
N_Gpts     = 2 * N_G    - 1
N_x        = 2 * N_Gpts - 1
dk         = (2 * np.pi) / (Kpoints * c)
G0         = (2 * np.pi) / c
q          = np.sqrt( k_choice[0]**2 + k_choice[1]**2 ) * dk
k          = np.zeros(Kpoints)
G          = np.zeros(N_G)
Gpts       = np.zeros(N_Gpts)
for ki in range (Kpoints): k[ki]    = (ki+0.5)*dk - np.pi/c
for Gi in range (N_G):     G[Gi]    = G0*(Gi-Gnum)
for Gi in range (N_Gpts):  Gpts[Gi] = G0*(Gi-2*Gnum)

#%%
#--------------------------------------------------------------------------    
#   This subroutine calculates the ALDA exchange kernel
#--------------------------------------------------------------------------
def fxc_calc(G2,nG):
    
    Uvec = np.zeros( N_x**2,          dtype=complex)
    RHS  = np.zeros( N_x**2,          dtype=complex)
    Pmat = np.zeros((N_x**2, N_x**2), dtype=complex)
    den  = np.zeros((N_x,    N_x),    dtype=complex) 
    FXC  = np.zeros((N_x,    N_x),    dtype=complex)
    G_R   = np.zeros( N_x )
    
    for Gi in range (N_x): G_R[Gi] = G0 * (Gi - 4 * Gnum)

    # define the sampling points in real space  
    x = y = np.zeros(N_x)
    dx = c / (N_x + 1)
    for i in range(N_x): x[i] = y[i] = -c/2 + dx*(i+1)

    # calculate the real-space density at the sampling points
    for xi in range(N_x):
        for yi in range(N_x):  
            density = 0
            for Gxi in range(N_Gpts):
                for Gyi in range(N_Gpts):
                    density += nG[Gxi,Gyi]                                     \
                               * cmath.exp(1j * (Gpts[Gxi]*x[xi]               \
                                              +  Gpts[Gyi]*y[yi]))
            den[xi,yi] = density.real 

    # the right-hand side are the LDA-x kernels at the sampling points
    if HXC==2:
        for i in range(N_x):
            for j in range(N_x):
                RHS[j + i*N_x] = -np.sqrt(2/(np.pi*den[i,j]))
                
    if HXC==3:
        a0 = -0.3568
        a1 = 1.13
        a2 = 0.9052
        a3 = 0.4165
        for i in range(N_x):
            for j in range(N_x):
                
                RHS[j + i*N_x] = -np.sqrt(2/(np.pi*den[i,j]))
                
                rs = 1/np.sqrt(np.pi*den[i,j])
                g = np.sqrt(rs)                
                
                g1 = 2*a1 + (3*a2+2*a1**2)*g + 0.75*(7*a3+5*a1*a2)*g**2 + 6*a1*a3*g**3
                
                g2 = 1.+a1*g+a2*g**2+a3*g**3
                
                g3 = 2. + 4*a1*g + (3*a2+2*a1**2)*g**2 + 0.5*(7*a3+5*a1*a2)*g**3\
                    +3*a1*a3*g**4
                    
                g4 = g2 * (a1 + 2*a2*g + 3*a3*g**2)
                
                RHS[j + i*N_x] -= (np.pi*a0/8.)*g**5*(g1*g2**2 - g3*g4)/g2**4
            
#  now define the coefficient matrix: we have
#  FX(x,y) = sum_{Gx,Gy} FX_{Gx,Gy}exp(-iGx*x-iGy*y)
    for i in range(N_x):
        for j in range(N_x):
            i1 = j + i*N_x
            for k in range(N_x):
                for l in range(N_x):
                    i2 = l + k*N_x            
                    Pmat[i1,i2] = cmath.exp(-1j*G_R[i]*x[k]-1j*G_R[j]*y[l])         

    Uvec = np.linalg.solve(Pmat,RHS) 
    
    for i in range(N_x):
        for j in range(N_x):
            FXC[i,j] = Uvec[j+i*N_x]

    return FXC

#%%---------------------------------------------------------------------------#
# This subroutine calculates the full response function "chi" from the Dyson  #
# equation. From "chi", the macroscopic dielectric function is then obtained. #  
#-----------------------------------------------------------------------------#
@njit() 
def chicalc(m3,m4,N_Gpts,Kp_x,Kp_y,Ntot,OccNum,E,MEGm,MEGp,iom,dk,c): 
    dum = 0.
    for kx in range(Kp_x):
        for ky in range(Kp_y):
            for j in range(Ntot):
                if OccNum[j,kx,ky]==1:
                    for l in range(Ntot):
                        if OccNum[l,kx,ky]==0:
                            de = E[j,kx,ky] - E[l,kx,ky]
                            dum += (MEGm[j,l,kx,ky,m3] \
                                   *MEGp[l,j,kx,ky,m4]/(de + iom)
                                  + MEGp[j,l,kx,ky,m4] \
                                   *MEGm[l,j,kx,ky,m3]/(de - iom))
                                                    
    chi0_out = -2.*c**2*dum*dk**2/(2*np.pi)**2 
    
    return chi0_out

#%%
def dy_calc(k_choice,FXC,om_pts,d_omega,eta,OccNum,Ntot,Kp_x,Kp_y,Gnum,\
            G2x,G2y,Cvec,dk,ME,E,c,quasi2D,mode,alpha_xc):
    
    m0    = 2*Gnum + 2*Gnum*N_G
    
    omega   = np.zeros(om_pts)
    epsilon = np.zeros(om_pts,                                dtype = complex)
    chi0    = np.zeros((N_Gpts**2,N_Gpts**2),                 dtype = complex)
    MEGp    = np.zeros((Ntot,Ntot,Kpoints,Kpoints,N_Gpts**2), dtype = complex)
    MEGm    = np.zeros((Ntot,Ntot,Kpoints,Kpoints,N_Gpts**2), dtype = complex)
    Mmat    = np.zeros((N_Gpts**2,N_Gpts**2),                 dtype = complex)
    RHS     = np.zeros( N_Gpts**2,                            dtype = complex)
    chi     = np.zeros( N_Gpts**2,                            dtype = complex)    
    Onemat  = np.zeros((N_Gpts**2, N_Gpts**2),                dtype = complex)
    VV      = np.zeros( N_Gpts**2)
    
    for m in range(N_Gpts*N_Gpts): Onemat[m,m] = 1
                         
    # first, calculate matrix elements
    for kx in range(Kp_x):
        for ky in range(Kp_y):
            for j in range(Ntot):
                for l in range(Ntot):
                    ol = OccNum[l,kx,ky]
                    oj = OccNum[j,kx,ky]
                    if (((ol == 0) and (oj==1))                                \
                    or  ((ol == 1) and (oj==0))):
                        for i1 in range(N_G):
                            for j1 in range(N_G): 
                                m1 = j1 + i1*N_G          # G1                  
                                for i2 in range(N_G):                                   
                                    for j2 in range(N_G):
                                        m2 = j2 + i2*N_G  # G2
                                            
                                        i3 = i1 - i2 + 2 * Gnum
                                        j3 = j1 - j2 + 2 * Gnum
                                        m3 = j3 + i3 * N_Gpts  
                                        
                                        i4 = i2-i1+2*Gnum
                                        j4 = j2-j1+2*Gnum
                                        m4 = j4 + i4*N_Gpts  
                                                
                                        if (m3 == m0):
                                            MEGm[j,l,kx,ky,m3] =               \
                                                -1j * ME[j,l,kx,ky]
                                        if (m4 == m0):
                                            MEGp[j,l,kx,ky,m4] =               \
                                                 1j * ME[j,l,kx,ky]
                                        else:  
                                            dd = np.conj(Cvec[j,kx,ky,m1])     \
                                               *         Cvec[l,kx,ky,m2]                                                         
                                            MEGm[j,l,kx,ky,m3] += dd                                    
                                            MEGp[j,l,kx,ky,m4] += dd
                                    
    for ii in range (om_pts):
        om = ii*d_omega
        omega[ii] = om
        iom = om + eta * 1j
                
#   calculate the noninteracting response function chi0_GG'(k-->0,omega)  
        for m3 in range(N_Gpts**2):
            for m4 in range(N_Gpts**2):
                chi0[m3,m4] = chicalc(m3,m4,N_Gpts,Kp_x,Kp_y,Ntot,OccNum,E,MEGm,MEGp,iom,dk,c)
                
        # now solve the Dyson equation        
        for i3 in range(N_Gpts*N_Gpts): RHS[i3] = chi0[i3,m0]
            
        if (mode == 12):    
            for i in range(N_Gpts):
                for j in range(N_Gpts):
                    m = j + i*N_Gpts
                    for i1 in range(N_Gpts):
                        for j1 in range(N_Gpts):
                            m1 = j1 + i1*N_Gpts 
                            if (m1 == m0):
                                v = 0.
                            else:
                                v = 2.*np.pi/np.sqrt(q**2 + Gpts[i1]**2 + Gpts[j1]**2)
                            Mmat[m,m1] = v*chi0[m,m1]      #modified Hartree
                            dum = 0
                            for i2 in range(N_Gpts):
                                for j2 in range(N_Gpts):
                                    m2 = j2 + i2*N_Gpts
                                    
                                    i4 = i2-i1+4*Gnum
                                    j4 = j2-j1+4*Gnum
                                    
                                    dum += chi0[m,m2]*FXC[i4,j4]/c**4
                                    
                            Mmat[m,m1] += dum
                                                       
        if ((mode == 13) or (mode == 14)): 
            
            for i4 in range(N_Gpts):
                for j4 in range(N_Gpts):
                    m4 = j4 + i4 * N_Gpts  
                    if m4==m0: 
                        VV[m4] = (2 * np.pi) / q
                    else:
                        VV[m4] = (2 * np.pi)                                   \
                                 / (np.sqrt(Gpts[i4]**2 + Gpts[j4]**2))
            if (mode == 14): VV[m0] = 0
                                                      
            for i3 in range(N_Gpts):
                for j3 in range(N_Gpts):
                    
                    m3 = j3 + i3*N_Gpts
                    
                    for i4 in range(N_Gpts):
                        for j4 in range(N_Gpts):
                            
                            m4 = j4 + i4*N_Gpts
                            
                            if m4==m0:
                                Mmat[m3,m4] = -(alpha_xc/(4*np.pi))*VV[m4]*q**2*chi0[m3,m4]
                            if m4!=m0:
                                Mmat[m3,m4] = -(alpha_xc/(4*np.pi))*VV[m4]*chi0[m3,m4]/c**2
                            
        Mmat += Onemat   

        chi = np.linalg.solve(Mmat,RHS) 

        if quasi2D == 0: epsilon[ii] = 1 - (2 * np.pi * q * chi[m0])
        if quasi2D == 1: epsilon[ii] = 1 - (4 * np.pi *     chi[m0]) / c**2
    
    return epsilon, omega   

