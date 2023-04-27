import math
import cmath
import numpy as np
#
def VHartree(HXC,Gnum,nG,G2):
    Mdim = 2*Gnum+1
    Mdim2 = 2*Mdim-1
    VHG2 = np.zeros((Mdim2,Mdim2), dtype=complex)
    VHG = np.zeros((Mdim,Mdim), dtype=complex)
    
    if HXC==1:
        for i in range(Mdim2):
            for j in range(Mdim2):   
                if i != Mdim-1 and j != Mdim-1:  # except G=0
                    VHG2[i,j]= 4*math.pi*nG[i,j]/(G2[i]**2+G2[j]**2)
#
#  In the line above, we obtain the Hartree potential in G-space:
#  VHG2 =  4 PI nG/G^2
#  We select an array of size Mdim*Mdim which goes into the Hamiltonian
#
        for i in range(Mdim):
            for j in range(Mdim):
                VHG[i,j] = VHG2[i+Gnum,j+Gnum]  
            
    return VHG        
#
#--------------------------------------------------------------------------    
#
def Vexchange(HXC,Gnum,nG,G,G2,c):
    
    Mdim = 2*Gnum+1
    Mdim2 = 2*Mdim-1
    Uvec = np.zeros(Mdim*Mdim, dtype=complex)
    RHS = np.zeros(Mdim*Mdim, dtype=complex)
    Pmat = np.zeros((Mdim*Mdim,Mdim*Mdim), dtype=complex)
    den = np.zeros((Mdim,Mdim)) 
    VXG = np.zeros((Mdim,Mdim), dtype=complex)
    ione = 0. + 1.j
    
    if HXC==1:
#
#  define the sampling points in real space  
#
        x = np.zeros(Mdim)
        y = np.zeros(Mdim)
        dx = c/(Mdim+1)
        for i in range(Mdim):
            x[i] = -c/2 + dx*(i+1)
        y=x.copy()    
#
#   calculate the real-space density at the sampling points
#
        for ii in range(Mdim):
            xx = x[ii]
            for jj in range(Mdim):  
                yy = y[jj]
                dum = 0.
                for i in range(Mdim2):
                    for j in range(Mdim2):
                        dum += nG[i,j]*cmath.exp(ione*(G2[i]*xx+G2[j]*yy))
                den[ii,jj] = dum.real 
#
#  the right-hand side are the LDA-x potentials at the sampling points
#
        for i in range(Mdim):
            for j in range(Mdim):
                VX = -2.*math.sqrt(2./math.pi)*math.sqrt(den[i,j])
                RHS[j + i*Mdim] = VX
#
#  now define the coefficient matrix: we have
#  V(x,y) = sum_{Gx,Gy} U_{Gx,Gy}exp(-iGx*x-iGy*y)
#
        for i in range(Mdim):
            for j in range(Mdim):
                i1 = j + i*Mdim
                for k in range(Mdim):
                    for l in range(Mdim):
                        i2 = l + k*Mdim            
                        Pmat[i1,i2] = cmath.exp(-ione*G[i]*x[k]-ione*G[j]*y[l])         

        Uvec = np.linalg.solve(Pmat,RHS) 
    
        for i in range(Mdim):
            for j in range(Mdim):
                if i != Gnum and j != Gnum:   # exclude G=0 (constant shift)
                    VXG[i,j] = Uvec[j+i*Mdim]

    return VXG

 