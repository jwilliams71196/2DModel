import numpy as np
from infile import *

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
    
N_G = 2*Gnum+1
Mdim2 = 2*N_G-1

C = np.zeros((Ntot,Kp_x,Kp_y,N_G*N_G), dtype=complex)
E = np.zeros((Ntot,Kp_x,Kp_y))

OccNum = np.zeros((Ntot,Kp_x,Kp_y))

nG  = np.zeros((Mdim2,Mdim2), dtype=complex)
nG0 = np.zeros((Mdim2,Mdim2), dtype=complex)
nG1 = np.zeros((Mdim2,Mdim2), dtype=complex)

VHG2    = np.zeros((Mdim2,Mdim2), dtype=complex)
VHG     = np.zeros((N_G,N_G), dtype=complex)
VHG_old = np.zeros((N_G,N_G), dtype=complex)
VXG     = np.zeros((N_G,N_G), dtype=complex)
VXG_old = np.zeros((N_G,N_G), dtype=complex)

dk = (np.pi/c)/Kpoints
G0 = 2*np.pi/c

kx  = np.zeros(Kp_x)
ky  = np.zeros(Kp_y)
G   = np.zeros(N_G)
G2  = np.zeros(Mdim2)

ME  = np.zeros((Ntot,Ntot,Kp_x,Kp_y),dtype=complex)

# Define k-point grid and G-points.
# We also need a G-point grid of twice the size, for nG

if Ksym==1 or Ksym==3:
    for i in range (Kpoints):
        kx[i] = (i+0.5)*dk
if Ksym==1 or Ksym==2:
    for i in range (Kpoints):
        ky[i] = (i+0.5)*dk
if Ksym==2 or Ksym==4:
    for i in range (2*Kpoints):
        kx[i] = (i+0.5)*dk - np.pi/c
if Ksym==3 or Ksym==4:
    for i in range (2*Kpoints):
        ky[i] = (i+0.5)*dk - np.pi/c
            
for j in range (N_G):
    G[j]  = G0*(j-Gnum)
for j in range (Mdim2):
    G2[j] = G0*(j-2*Gnum)
    
C0 = np.zeros((Ntot,Kp_x,Kp_y,N_G*N_G), dtype=complex)
Ct = np.zeros((Ntot,Kp_x,Kp_y,N_G*N_G), dtype=complex)
Ct1 = np.zeros((Ntot,Kp_x,Kp_y,N_G*N_G), dtype=complex)
RHS = np.zeros((N_G*N_G), dtype=complex)
Mmat = np.zeros((N_G*N_G,N_G*N_G), dtype=complex)
ONE = np.zeros((N_G*N_G,N_G*N_G))
PHI = np.zeros((N_G*N_G), dtype=complex)
Nex = np.zeros(Tsteps)
Epsilon_t = np.zeros(Tsteps)
Dipole_t = np.zeros(Tsteps)
Time = np.zeros(Tsteps)
jx_mac = np.zeros(Tsteps)
jy_mac = np.zeros(Tsteps)
Axc_x = np.zeros(Tsteps)
Axc_y = np.zeros(Tsteps)
drive = np.zeros(Tsteps)
F_occ = np.zeros(Kp_x*Kp_y*Ntot)

#   the name of the folder that the TD calculation results get written to
location  = "./Results/E0=" + str(E0) + "alpha_xc=" + str(alpha_xc) + "/"