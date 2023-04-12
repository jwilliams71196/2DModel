from infile import *
from initializations import *
from mat_def import *
#from dencalc import Density

import numpy as np
import matplotlib.pyplot as plt
import cmath
import sys

# --------------------------------------------------------------------------- #
#    This function calculates the ground state band structure and energies    #
# --------------------------------------------------------------------------- #
def GroundState():

    # ----------------------------------------------------------------------- #
    #                     START OF SELF-CONSISTENCY LOOP                      #
    #                  (solve for each k-point independently)                 #
    # ----------------------------------------------------------------------- #
    Eref = 1.0
    counter=0

    Eaux = np.zeros(Kp_x*Kp_y*Ntot)

    VHG2    = np.zeros((Mdim2,Mdim2), dtype=complex)
    VHG     = np.zeros((N_G,N_G), dtype=complex)
    VHG_old = np.zeros((N_G,N_G), dtype=complex)
    VXG     = np.zeros((N_G,N_G), dtype=complex)
    VXG_old = np.zeros((N_G,N_G), dtype=complex)

    T_mat = np.zeros((N_G*N_G,N_G*N_G), dtype=complex)
    U_mat = np.zeros((N_G*N_G,N_G*N_G), dtype=complex)
    H_mat = np.zeros((N_G*N_G,N_G*N_G), dtype=complex)

    while abs(Eref-E[1,0,0])>TOL:
        counter+=1
        print("Iteration no.", counter, "criterion = ",abs(Eref-E[1,0,0]))
        Eref = E[1,0,0]

        U_mat = U_mat_def(N_G,Gnum,A,B,VHG,VXG)

        for kxi in range(Kp_x):
            for kyi in range(Kp_y):
                T_mat = T_mat_k_def(kx[kxi],ky[kyi],G)

                H_mat = T_mat + U_mat

                vals, vecs = np.linalg.eigh(H_mat)
                for n in range(Ntot):
                    E[n,kxi,kyi] = vals[n]
                    Eaux[n + kyi*Ntot + kxi*Kp_y*Ntot]=E[n,kxi,kyi]
                    for Gi in range (N_G**2):
                        C[n,kxi,kyi,Gi] = vecs[Gi,n]

#       find the Fermi energy and the occupation numbers
        sorted = np.argsort(Eaux)
        EF_index = Kp_x*Kp_y*Nel//2 - 1
        EF = Eaux[sorted[EF_index]]
        print("Fermi energy:",round(EF,5))
        for kxi in range(Kp_x):
            for kyi in range(Kp_y):
                for n in range(Ntot):
                    if E[n,kxi,kyi]<=EF:
                        OccNum[n,kxi,kyi] = 1
                    else:
                        OccNum[n,kxi,kyi] = 0

#       find the band gap
        Eg = sys.maxsize
        for kxi in range(Kpoints):
            for kyi in range(Kpoints):
                Omega = E[Nel//2,kxi,kyi] - E[Nel//2-1,kxi,kyi]
                if Omega < Eg: Eg = Omega
        print("Band gap:",round(Eg,5))

        nG = Density(Ksym,OccNum,Ntot,Kp_x,Kp_y,Gnum,dk,C,G,c,HXC,GSout)

        if HXC == 1:
            VHG_old = VHG.copy()
            VHG = V_Hartree_mat_def(HXC,Gnum,nG,G2)
            VHG = MIX*VHG + (1.0-MIX)*VHG_old

            VXG_old = VXG.copy() 
            VXG = V_Exchange_mat_def(HXC,Gnum,nG,G,G2,c)
            VXG = MIX*VXG + (1.0-MIX)*VXG_old

        elif HXC == 0: Eref = E[1,0,0]   # because in this case we want only 1 iteration,
                                         # and this will exit the loop

        else:
            print('Error: HXC must be either 0 or 1 (in infile).')
            end_time = time.time()
            print("Exited after " + str(end_time - start_time) + "s")
            sys.exit()
    # ----------------------------------------------------------------------- #
    #                     END OF SELF-CONSISTENCY LOOP                        #
    # ----------------------------------------------------------------------- #
    if GSout==1: den_diag_out(c,Mdim2,nG,G2)
    if GSout==2: den_full_out(c,Mdim2,nG,G2)
    if GSout==3: band_out(c,N_G,U_mat,G,EF)

    return E, C, Eg, nG, OccNum

# --------------------------------------------------------------------------- #
#    This function calculates the dipole matrix elements for later use        #
# --------------------------------------------------------------------------- #
def DipoleMatrixElements(OccNum,C,E):
    #   Calculate the dipole matrix elements ME (along the x-direction).
    #   Notice that we want to avoid matrix elements between near-degenerate
    #   bands. To do this we demand that the energies are separated by more than 0.1,
    #   which happens here (*). The fudge factor 0.1 will need to be explained later.

    for i in range (Kp_x):
        for j in range (Kp_y):
            for l in range (Ntot):
                for m in range (Ntot):
                    if OccNum[m,i,j] != OccNum[l,i,j] or abs(m-l)>1:
                        temp = 0.0
                        for G1i in range (N_G):
                            for G2i in range (N_G):
                                ind = G2i + G1i*N_G
                                temp += np.conj(C[l,i,j,ind])*C[m,i,j,ind]*G[G1i]
                        ME[l,m,i,j] = temp/(E[l,i,j] - E[m,i,j])   
 
                    elif abs(m-l)==1 and abs(E[l,i,j] - E[m,i,j])>0.1:  #(*)
                        temp = 0.0
                        for k1 in range (N_G):
                            for k2 in range (N_G):
                                ind = k2 + k1*N_G
                                temp += np.conj(C[l,i,j,ind])*C[m,i,j,ind]*G[k1]
                        ME[l,m,i,j] = temp/(E[l,i,j] - E[m,i,j])

    return ME

# --------------------------------------------------------------------------- #
#    This function calculates the frequency-dependent dielectric function     #
# --------------------------------------------------------------------------- #
def EpsilonOmega(OccNum,ME,E,Eg):

    omega = np.zeros(omegavalues)
    epsilon = np.zeros(omegavalues, dtype=complex)
    chi_k = np.zeros((Kp_x,Kp_y), dtype=complex)

    for omega_i in range(omegavalues):
        om = omega_i*d_omega
        omega[omega_i] = om
        if omega_i%10==0: print(omega_i,'\tout of',omegavalues)

        for kxi in range(Kp_x):
            for kyi in range(Kp_y):
                temp = 0.0
                for n in range(Ntot):
                    if OccNum[n,kxi,kyi]==1:
                        for m in range(Ntot):
                            if OccNum[m,kxi,kyi]==0:
                                temp += abs(ME[n,m,kxi,kyi])**2  \
                                   *( 1/(E[n,kxi,kyi] - E[m,kxi,kyi] + om + 1j*eta)\
                                     +1/(E[n,kxi,kyi] - E[m,kxi,kyi] - om - 1j*eta) )
                chi_k[kxi,kyi] = temp

        chi = 0.0
        for kxi in range(Kp_x):
            for kyi in range(Kp_y):
                chi += chi_k[kxi,kyi] 

        chi = ((2/c**2)*chi*(dk**2)) / ((2*np.pi)**2)
#               ^ factor of 2 because of spin
        if Ksym==1:
            chi= 4*chi
        if Ksym==2 or Ksym==3:
            chi = 2*chi

        if quasi2D == 0: epsilon[omega_i] = 1 - (2*np.pi/kval)*(kval**2)*chi
        if quasi2D == 1: epsilon[omega_i] = 1 - (4*np.pi)*chi

    epsilon00 = epsilon[0].real
    print()
    print('epsilon00 = ' + str(epsilon00))
    plt.plot(omega,epsilon.real,label='Real')
    plt.plot(omega,epsilon.imag,label='Imaginary')
    plt.axvline(x=Eg, color='black', label='Band Gap', ls=':')
    plt.title('Frequency Dependent Dielectric Function')
    plt.xlabel('Frequency (Energy)')
    plt.ylabel('Dielectric Function')
    plt.xlim([0,(omegavalues-1)*d_omega])
    plt.legend()
    plt.show() 

    filename = "eps_omega.txt"

    f = open(filename, "w")
    for omega_i in range(omegavalues):
        f.write(str(round(omega[omega_i],5)) + "  "\
                + str(round(epsilon.real[omega_i],5)) + "  "\
                + str(round(epsilon.imag[omega_i],5)) +'\n')
    f.close()

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ CALCULATIONS ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ #
#---------------------------------------------------------------------------- #
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv OUTPUTS vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv #

# --------------------------------------------------------------------------- #
#    This function plots the density along the diagonal of the BZ             #
# --------------------------------------------------------------------------- #
def den_diag_out(c,Mdim2,nG,G2x,G2y):

#  Calculate and plot the real space density along the diagonal

    ione = 0. + 1.j
    Xpoints = 41
    x = np.zeros(Xpoints)
    den = np.zeros(Xpoints)
    dx = c/(Xpoints-1)
    for ii in range(Xpoints):
        xx = -c/2 + ii*dx
        yy = xx
        x[ii] = xx

        dum = 0.0
        for i in range(Mdim2):
            for j in range(Mdim2):
                dum += nG[i,j]*cmath.exp(ione*(G2x[i]*xx+G2y[j]*yy))
        den[ii] = dum.real
    plt.plot(x,den)
    plt.show()

# --------------------------------------------------------------------------- #
#    This function plots the full 2D density                                  #
# --------------------------------------------------------------------------- #
def den_full_out(c,Mdim2,nG,G2x,G2y):

#  Calculate and plot the 2D real space density

    Xpoints = 31
    x = np.zeros(Xpoints)
    den = np.zeros((Xpoints,Xpoints))
    dx = c/(Xpoints-1)
    for ii in range(Xpoints):
        xx = -c/2 + ii*dx
        x[ii] = xx
        for jj in range(Xpoints):
            yy = -c/2 + jj*dx
            dum = 0.0
            for i in range(Mdim2):
                for j in range(Mdim2):
                    dum += nG[i,j]*cmath.exp(1j*(G2x[i]*xx+G2y[j]*yy))
            den[ii,jj] = dum.real

    y = x.copy()
    X,Y = np.meshgrid(x,y)
    fig = plt.figure(figsize =(14, 9))
    ax = plt.axes(projection ='3d')
    ax.plot_surface(X, Y, den)
    plt.show()

# --------------------------------------------------------------------------- #
#    This function plots the ground state band structure                      #
# --------------------------------------------------------------------------- #
def band_out(c,Mdim,Umat,G,EF):

#  Calculate band structure along Gamma --> X --> M --> Gamma in the BZ

#   Nbands = 5
#   Kpoints = 51
    EB = np.zeros((Nbands,3*Kpoints-2))
    dk = (np.pi/c)/(Kpoints-1)
    kx = np.zeros((3*Kpoints-2))
    ky = np.zeros((3*Kpoints-2))
    Tmat = np.zeros((N_G**2,N_G**2), dtype=complex)
    for ki in range (Kpoints):
        kx[ki] = ki*dk
        ky[ki] = 0.0
    for ki in range (Kpoints-1):
        kx[ki+Kpoints] = np.pi/c
        ky[ki+Kpoints] = (ki+1)*dk
    for ki in range (Kpoints-1):
        kx[ki+2*Kpoints-1] = np.pi/c - (ki+1)*dk
        ky[ki+2*Kpoints-1] = np.pi/c - (ki+1)*dk

    for ki in range (3*Kpoints-2):
        for Gix in range(N_G):
            for Giy in range(N_G):
                Tmat[Giy + Gix*N_G,Giy + Gix*N_G] = 0.5*( (kx[ki]-G[Gix])**2 + (ky[ki]-G[Giy])**2 )

        Hmat = Tmat + Umat
        vals, vecs = np.linalg.eigh(Hmat)
        for n in range(Nbands):
            EB[n,ki] = vals[n]

    EB = EB-EF

    for n in range(Nbands):
        plt.plot(EB[n])
#   plt.plot(EB[0])
#   plt.plot(EB[1])
#   plt.plot(EB[2])
#   plt.plot(EB[3])
#   plt.plot(EB[4])

    plt.xlabel("k")
    plt.ylabel("E")
    plt.title('$\Gamma$                         X                          M                         $\Gamma$') # lol
    plt.tick_params(labelbottom = False, bottom = False)

#find the minimum and maximum of each band

    Emm = np.zeros((2*Nbands))
    kmm = np.zeros((2*Nbands))
    for i in range(Nbands):
        m1 = EB[i,0]
        m2 = EB[i,0]
        km1=0
        km2=0
        for j in range (3*Kpoints-2):
            if (EB[i,j]>m2):
                m2=EB[i,j]
                km2=j
            if (EB[i,j]<m1):
                m1=EB[i,j]
                km1=j
        Emm[2*i]=m1
        Emm[2*i+1]=m2
        kmm[2*i]=km1
        kmm[2*i+1]=km2

    print()

    round_num = 5

    for n in range(0,Nbands,2):
        if Emm[n] >= 0 and Emm[n+1] >= 0:
            print("min/max of Band", int(n/2+2), ":", round(Emm[n],round_num), round(Emm[n+1],round_num))
        elif Emm[n] >= 0 and Emm[n+1] < 0:
            print("min/max of Band", int(n/2+2), ":", round(Emm[n],round_num), round(Emm[n+1],round_num))
        elif Emm[n] < 0 and Emm[n+1] >= 0:
            print("min/max of Band", int(n/2+2), ":", round(Emm[n],round_num), round(Emm[n+1],round_num))
        else:
            print("min/max of Band", int(n/2+2), ":", round(Emm[n],round_num), round(Emm[n+1],round_num))

    print()

    plt.plot(kmm,Emm,'o')
    Eout = np.zeros(2)
    EFout = np.zeros(2)
#    EFout[0]=EF
#    EFout[1]=EF
    Eout[0]=0
    Eout[1]=150
    plt.plot(Eout,EFout,linestyle='dashed',color='black')
    plt.show()