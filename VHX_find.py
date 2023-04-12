import cmath
import numpy as np

def VHartree(HXC,Gnum,nG,G2):
    N_G   = 2*Gnum+1
    Mdim2 = 2*N_G-1

    VHG  = np.zeros((N_G,N_G), dtype=complex)
    VHG2 = np.zeros((Mdim2,Mdim2), dtype=complex)

    if HXC==1:
        for i in range(Mdim2):
            for j in range(Mdim2):   
                if i != N_G-1 and j != N_G-1:  # except G=0
                    VHG2[i,j]= 4*np.pi*nG[i,j]/(G2[i]**2+G2[j]**2)

#  In the line above, we obtain the Hartree potential in G-space:
#  VHG2 =  4 PI nG/G^2
#  We select an array of size N_G*N_G which goes into the Hamiltonian

        for Gxi in range(N_G):
            for Gyi in range(N_G):
                VHG[Gxi,Gyi] = VHG2[Gxi+Gnum,Gyi+Gnum]

    return VHG

#--------------------------------------------------------------------------    

def Vexchange(HXC,Gnum,nG,G,G2,c):
    
    N_G = 2*Gnum+1
    Mdim2 = 2*N_G-1
    Uvec = np.zeros(N_G*N_G, dtype=complex)
    RHS = np.zeros(N_G*N_G, dtype=complex)
    Pmat = np.zeros((N_G*N_G,N_G*N_G), dtype=complex)
    den = np.zeros((N_G,N_G))
    VXG = np.zeros((N_G,N_G), dtype=complex)
    ione = 0. + 1.j

    if HXC==1:
#  define the sampling points in real space

        x_G = np.zeros(N_G)
        dx = c/(N_G+1)
        for Gxi in range(N_G):
            x_G[Gxi] = -c/2 + dx*(Gxi+1)
        y_G=x_G.copy()

#   calculate the real-space density at the sampling points
        for Gxi in range(N_G):
            x = x_G[Gxi]
            for Gyi in range(N_G):
                y = y_G[Gyi]
                dum = 0.0
                for i in range(Mdim2):
                    for j in range(Mdim2):
                        dum += nG[i,j]*cmath.exp(1j*(G2[i]*x+G2[j]*y))
                den[Gxi,Gyi] = dum.real 

#  the right-hand side are the LDA-x potentials at the sampling points
        for Gxi in range(N_G):
            for Gyi in range(N_G):
                VX = -2.*np.sqrt(2./np.pi)*np.sqrt(den[Gxi,Gyi])
                RHS[Gyi + Gxi*N_G] = VX

#  now define the coefficient matrix: we have
#  V(x,y) = sum_{Gx,Gy} U_{Gx,Gy}exp(-iGx*x-iGy*y)
        for Gxi1 in range(N_G):
            for Gyi1 in range(N_G):
                Gi1 = Gyi1 + Gxi1*N_G
                for Gxi2 in range(N_G):
                    for Gyi2 in range(N_G):
                        Gi2 = Gyi2 + Gxi2*N_G
                        Pmat[Gi1,Gi2] = cmath.exp(-1j*G[Gxi1]*x_G[Gxi2]-1j*G[Gyi1]*y_G[Gyi2])

        Uvec = np.linalg.solve(Pmat,RHS) 

        for Gxi in range(N_G):
            for Gyi in range(N_G):
                if Gxi != Gnum and Gyi != Gnum:   # exclude G=0 (constant shift)
                    VXG[Gxi,Gyi] = Uvec[Gyi+Gxi*N_G]

    return VXG
