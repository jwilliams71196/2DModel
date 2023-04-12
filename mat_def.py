from infile import *
from initializations import *

import numpy as np

import cmath

#-----------------------------------------------------------------------------#
#   U_mat_def defines the scalar potential                                    #
#-----------------------------------------------------------------------------#
def U_mat_def(N_G,Gnum,A,B,VHG,VXG):

    U     = np.zeros((N_G,N_G), dtype=complex)
    U_mat = np.zeros((N_G**2,N_G**2), dtype=complex)

    for Gxi in range(-Gnum,Gnum+1):
        for Gyi in range(-Gnum,Gnum+1):
            if abs(Gxi) == 0 and abs(Gyi) == 1:
                U[Gxi,Gyi] = -(A-B)/2.0
            if abs(Gxi) == 1 and abs(Gyi) == 0:
                U[Gxi,Gyi] = -(A-B)/2.0
            if abs(Gxi) == 1 and abs(Gyi) == 1:
                U[Gxi,Gyi] = -(A+B)/4.0

    U += VHG
    U += VXG

    for Gxi1 in range(-Gnum,Gnum+1):
        for Gyi1 in range(-Gnum,Gnum+1):
            for Gxi2 in range(-Gnum,Gnum+1):
                for Gyi2 in range(-Gnum,Gnum+1):

                    dGx = Gxi2 - Gxi1
                    dGy = Gyi2 - Gyi1

                    if (abs(dGx)<=Gnum and abs(dGy)<=Gnum):
                        U_mat[Gxi + Gyi*N_G,Gxi2 + Gyi2*N_G] = U[dGx+Gnum,dGy+Gnum]
    return U_mat

#-----------------------------------------------------------------------------#
#   T_mat_k_def defines the kinetic energy at a given k-point                 #
#-----------------------------------------------------------------------------#
def T_mat_k_def(kx,ky,G):

    T_mat_k = np.zeros((N_G*N_G,N_G*N_G), dtype=complex)

    for Gxi in range(N_G):
        for Gyi in range(N_G):
            T_mat_k[Gyi + Gxi*N_G,Gyi + Gxi*N_G] = 0.5*((kx-G[Gxi])**2 + (ky-G[Gyi])**2)

    return T_mat_k

#-----------------------------------------------------------------------------#
#   V_Hartree_mat_def_def defines the Hartree energy                          #
#-----------------------------------------------------------------------------#
def V_Hartree_mat_def(HXC,Gnum,nG,G2):
#    N_G   = 2*Gnum+1
#    Mdim2 = 2*N_G-1

    VHG  = np.zeros((N_G,N_G), dtype=complex)
    VHG2 = np.zeros((Mdim2,Mdim2), dtype=complex)

    if HXC==1:
        for i in range(Mdim2):
            for j in range(Mdim2):
                if i != N_G-1 and j != N_G-1: # except G=0
                    VHG2[i,j]= 4*np.pi*nG[i,j]/(G2[i]**2+G2[j]**2)

#  In the line above, we obtain the Hartree potential in G-space:
#  VHG2 =  4 PI nG/G^2
#  We select an array of size N_G*N_G which goes into the Hamiltonian

        for Gxi in range(N_G):
            for Gyi in range(N_G):
                VHG[Gxi,Gyi] = VHG2[Gxi+Gnum,Gyi+Gnum]

    return VHG

#-----------------------------------------------------------------------------#
#   V_Exchange_mat_def defines the exchange energy                            #
#-----------------------------------------------------------------------------#
def V_Exchange_mat_def(HXC,Gnum,nG,G,G2,c):
    
    N_G   = 2*Gnum+1
    Mdim2 = 2*N_G-1
    
    Uvec  = np.zeros(N_G*N_G, dtype=complex)
    RHS   = np.zeros(N_G*N_G, dtype=complex)
    P_mat = np.zeros((N_G*N_G,N_G*N_G), dtype=complex)
    den   = np.zeros((N_G,N_G))
    VXG   = np.zeros((N_G,N_G), dtype=complex)

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
                temp = 0.0
                for i in range(Mdim2):
                    for j in range(Mdim2):
                        temp += nG[i,j]*cmath.exp(1j*(G2[i]*x+G2[j]*y))
                den[Gxi,Gyi] = temp.real 

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
                        P_mat[Gi1,Gi2] = cmath.exp(-1j*G[Gxi1]*x_G[Gxi2]-1j*G[Gyi1]*y_G[Gyi2])

        Uvec = np.linalg.solve(P_mat,RHS) 

        for Gxi in range(N_G):
            for Gyi in range(N_G):
                if Gxi != Gnum and Gyi != Gnum:   # exclude G=0 (constant shift)
                    VXG[Gxi,Gyi] = Uvec[Gyi+Gxi*N_G]

    return VXG

#-----------------------------------------------------------------------------#
#   A_mat_def defines the time-dependent vector potential                     #
#-----------------------------------------------------------------------------#
def A_mat_def(N_G,Gnum,C):
    AVx = np.zeros((N_G,N_G), dtype=complex)
    AVy = np.zeros((N_G,N_G), dtype=complex)
    
    Ax_mat = np.zeros((N_G*N_G,N_G*N_G), dtype=complex)
    Ay_mat = np.zeros((N_G*N_G,N_G*N_G), dtype=complex)
    A2_mat = np.zeros((N_G*N_G,N_G*N_G), dtype=complex)

    for Gxi in range(-Gnum,Gnum+1):
        for Gyi in range(-Gnum,Gnum+1):
            if Gxi==0 and abs(Gyi)==1:
                AVy[Gxi,Gyi] = np.pi*C*Gyi*1j
            if Gyi==0 and abs(Gxi)==1:
                AVx[Gxi,Gyi] = np.pi*C*Gxi*1j
            if abs(Gxi)==1 and abs(Gyi)==1:
                AVx[Gxi,Gyi] = 0.5*np.pi*C*Gxi*1j
                AVy[Gxi,Gyi] = 0.5*np.pi*C*Gyi*1j

    for Gxi1 in range(-Gnum,Gnum+1):
        for Gyi1 in range(-Gnum,Gnum+1):
            for Gxi2 in range(-Gnum,Gnum+1):
                for Gyi2 in range(-Gnum,Gnum+1):

                    dGx = Gxi2 - Gxi1
                    dGy = Gyi2 - Gyi1

                    if (abs(dGx)<=Gnum and abs(dGy)<=Gnum):
                        Ax_mat[Gxi1 + Gyi1*N_G,Gxi2 + Gyi2*N_G] = AVx[dGx+Gnum,dGy+Gnum]
                        Ay_mat[Gxi1 + Gyi1*N_G,Gxi2 + Gyi2*N_G] = AVy[dGx+Gnum,dGy+Gnum]

                    for Gxi3 in range(-Gnum,Gnum+1):
                        for Gyi3 in range(-Gnum,Gnum+1):
                            
                            dG23x = Gxi2 - Gxi3
                            dG21x = Gxi2 - Gxi1
                            dG23y = Gyi2 - Gyi3
                            dG21y = Gyi3 - Gyi1
                            if (abs(dG23x)<=Gnum and abs(dG23y)<=Gnum \
                                and abs(dG21x)<=Gnum and abs(dG21y)<=Gnum):
                                    A2_mat[Gxi1 + Gyi1*N_G,Gxi2 + Gyi2*N_G] += 0.5*\
                           (AVx[dG23x+Gnum,dG23y+Gnum]*AVx[dG21x+Gnum,dG21y+Gnum] +\
                            AVy[dG23x+Gnum,dG23y+Gnum]*AVy[dG21x+Gnum,dG21y+Gnum])

    return Ax_mat,Ay_mat,A2_mat
