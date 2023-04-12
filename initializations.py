from infile import *

import numpy as np

if Ksym==1:
    Kp_x = Kpoints
    Kp_y = Kpoints
if Ksym==2:
    Kp_x = 2*Kpoints
    Kp_y = Kpoints
if Ksym==3:
    Kp_x = Kpoints
    Kp_y = 2*Kpoints
if Ksym==4:
    Kp_x = 2*Kpoints
    Kp_y = 2*Kpoints
    
N_G   = 2*Gnum+1
Mdim2 = 2*N_G-1

C = np.zeros((Ntot,Kp_x,Kp_y,N_G*N_G), dtype=complex)
E = np.zeros((Ntot,Kp_x,Kp_y))

OccNum = np.zeros((Ntot,Kp_x,Kp_y))

nG  = np.zeros((Mdim2,Mdim2), dtype=complex)
nG0 = np.zeros((Mdim2,Mdim2), dtype=complex)
nG1 = np.zeros((Mdim2,Mdim2), dtype=complex)

dk = (np.pi/c)/Kpoints

kx  = np.zeros(Kp_x)
ky  = np.zeros(Kp_y)

G   = np.zeros(N_G)
G2  = np.zeros(Mdim2)

ME  = np.zeros((Ntot,Ntot,Kp_x,Kp_y),dtype=complex)

# Define k-point grid and G-points.

for ki in range (Kpoints):
    if Ksym==1 or Ksym==3: kx[ki] = (ki+0.5)*dk
    if Ksym==2 or Ksym==4: kx[ki] = (ki+0.5)*dk - np.pi/c
    if Ksym==1 or Ksym==2: ky[ki] = (ki+0.5)*dk
    if Ksym==3 or Ksym==4: ky[ki] = (ki+0.5)*dk - np.pi/c

for Gi in range(-Gnum,Gnum+1):
    G[Gi]  = 2*np.pi*(Gi-Gnum)/c
# We also need a larger G-point grid for nG (the density)
for j in range (Mdim2):
    G2[j] = 2*np.pi**(j-2*Gnum)/c

C0  = np.zeros((Ntot,Kp_x,Kp_y,N_G*N_G), dtype=complex)
Ct  = np.zeros((Ntot,Kp_x,Kp_y,N_G*N_G), dtype=complex)
Ct1 = np.zeros((Ntot,Kp_x,Kp_y,N_G*N_G), dtype=complex)

RHS   = np.zeros((N_G*N_G), dtype=complex)
M_mat = np.zeros((N_G*N_G,N_G*N_G), dtype=complex)

Identity = np.zeros((N_G**2,N_G**2))
PHI      = np.zeros((N_G**2), dtype=complex)

Nex       = np.zeros(Tsteps)
Epsilon_t = np.zeros(Tsteps)
Dipole_t  = np.zeros(Tsteps)
Time      = np.zeros(Tsteps)
jx_mac    = np.zeros(Tsteps)
jy_mac    = np.zeros(Tsteps)
Axc_x     = np.zeros(Tsteps)
Axc_y     = np.zeros(Tsteps)
drive     = np.zeros(Tsteps)
F_occ     = np.zeros(Kp_x*Kp_y*Ntot)

# the name of the folder that the TD calculation results get written to
location  = "./Results/E0=" + str(E0) + "alpha_xc=" + str(alpha_xc) + "/"

# ---------------------------- GENERAL FUNCTIONS ---------------------------- #

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