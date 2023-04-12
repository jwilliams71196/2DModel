from infile import *

import numpy as np

#------------------------------------------------------------------------------
#    This subroutine calculates the density, nG, in G-space
#    using occupation numbers and simple integration
#------------------------------------------------------------------------------
def Density(Ksym,OccNum,Ntot,Kp_x,Kp_y,Gnum,dk,C,G,c,HXC,out):

    N_G = 2*Gnum+1
    Mdim2 = 2*N_G-1

    nG = np.zeros((Mdim2,Mdim2), dtype=complex)
    nG1 = np.zeros((Mdim2,Mdim2), dtype=complex)
    nGk = np.zeros((2*N_G,2*N_G,Kp_x,Kp_y), dtype=complex)

    if HXC==1 or out==1 or out==2:

        for kxi in range(Kp_x):
            for kyi in range(Kp_y):
                for n in range(Ntot):
                    if OccNum[n,kxi,kyi]==1:
                        for Gxi1 in range(N_G):
                            for Gyi1 in range(N_G):
                                Gi1 = Gyi1 + Gxi1*N_G
                                for Gxi2 in range(N_G):
                                    for Gyi2 in range(N_G):
                                        Gi2 = Gyi2 + Gxi2*N_G

                                        Gxi3 = Gxi2-Gxi1+2*Gnum
                                        Gyi3 = Gyi2-Gyi1+2*Gnum

                                        nGk[Gxi3,Gyi3,kxi,kyi] += C[n,kxi,kyi,Gi1]*np.conj(C[n,kxi,kyi,Gi2])

#   Now integrate over the Brillouin zone

        for i in range(Mdim2):
            for j in range(Mdim2):

                for kxi in range(Kp_x):
                    for kyi in range(Kp_y):
                        nG1[i,j] += nGk[i,j,kxi,kyi]

                nG1[i,j] = 2.*nG1[i,j]*dk**2/(2*np.pi)**2

        for i in range(Mdim2):
            for j in range(Mdim2):
                if Ksym==1:
                    mm = Mdim2-1
                    nG[i,j] = nG1[i,j] + nG1[mm-i,j] + nG1[i,mm-j] + nG1[mm-i,mm-j]
                if Ksym==2:
                    mm = Mdim2-1
                    nG[i,j] = nG1[i,j] + nG1[i,mm-j]
                if Ksym==3:
                    mm = Mdim2-1
                    nG[i,j] = nG1[i,j] + nG1[mm-i,j]
                if Ksym==4:
                    nG[i,j] = nG1[i,j]

    return nG
#------------------------------------------------------------------------------
#    This subroutine calculates the macroscopic paramagnetic current density
#------------------------------------------------------------------------------
def CurrentDensity(Ksym,OccNum,Ntot,Kp_x,Kp_y,N_G,dk,Ct,Gx,Gy,kx,ky):

    jx = 0.0
    jy = 0.0

    jx_k = np.zeros((Kp_x,Kp_y))
    jy_k = np.zeros((Kp_x,Kp_y))

    for kxi in range (Kp_x):
        for kyi in range (Kp_y):
            for n in range (Ntot):
                if OccNum[n,kxi,kyi]==1:
                    for Gi1 in range(N_G):
                        for Gi2 in range(N_G):
                            m = Gi2 + Gi1*N_G
                            jx_k[kxi,kyi] += Gx[Gi1]*np.abs(Ct[n,kxi,kyi,m])**2
                            jy_k[kxi,kyi] += Gy[Gi2]*np.abs(Ct[n,kxi,kyi,m])**2

#   Now integrate over the Brillouin zone

    for kxi in range(Kp_x):
        for kyi in range(Kp_y):
            jx += jx_k[kxi,kyi]
            jy += jy_k[kxi,kyi]

    jx = 2*jx*dk**2/(2*np.pi)**2
    jy = 2*jy*dk**2/(2*np.pi)**2
    jy = 0.0

    if Ksym==2 or Ksym==3:
        jx = 2*jx
        jy = 2*jy

        return jx,jy
#------------------------------------------------------------------------------
#    This subroutine calculates the number of excited electrons
#------------------------------------------------------------------------------
def ExcitedStatePopulation(Ksym,OccNum,Ntot,Kp_x,Kp_y,N_G,dk,C,Ct,c):

    N_ex = 0.0
    
    nk = np.zeros((Kp_x,Kp_y))

    for kxi in range (Kp_x):
        for kyi in range (Kp_y):
            for n in range (Ntot):
                if OccNum[n,kxi,kyi]==1:
                    for m in range(Ntot):
                        if OccNum[m,kxi,kyi]==0:
                            temp = 0.0
                            for Gi in range (N_G**2):
                                temp += Ct[n,kxi,kyi,Gi]*np.conj(C[m,kxi,kyi,Gi])
                            nk[kxi,kyi] +=  np.abs(temp)**2

#   Now integrate over the Brillouin zone

    for kxi in range(Kp_x):
        for kyi in range(Kp_y):
            N_ex += nk[kxi,kyi]

    N_ex = 2.*N_ex*dk**2/(2*np.pi)**2    # factor 2 because of spin

    if Ksym==1:
        N_ex= 4*N_ex
    if Ksym==2 or Ksym==3:
        N_ex = 2*N_ex

    return N_ex*c**2
#------------------------------------------------------------------------------
#    This function calculates the occupation numbers as a function
#    of energy.
#------------------------------------------------------------------------------
def FermiDistance(Ksym,OccNum,Ntot,Kp_x,Kp_y,N_G,dk,C,Ct):

#    Faux = np.zeros((Ntot,Kp_x,Kp_y))
    F_occ = np.zeros(Kp_x*Kp_y*Ntot)

    for kxi in range (Kp_x):
        for kyi in range (Kp_y):
            for n in range (Ntot):
                for m in range(Ntot):
                    if OccNum[m,kxi,kyi]==1:
                        temp = 0.0
                        for Gi in range (N_G**2):
                            temp += Ct[m,kxi,kyi,Gi]*np.conj(C[n,kxi,kyi,Gi])
#                        Faux[n,kxi,kyi] +=  np.abs(temp)**2
                        F_occ[n + kyi*Ntot + kxi*Kp_y*Ntot] += np.abs(temp)**2

    return F_occ
