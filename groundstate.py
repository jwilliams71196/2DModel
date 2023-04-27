from infile import *
from initializations import *
from Band_output import *

def GroundState():
    
    from mat_def import Umat_def
    from dencalc import Density
    from VHX_find import VHartree, Vexchange
    import numpy as np
    import matplotlib.pyplot as plt
    import sys
#-----------------------------------------------------------------------------#
#                       START OF SELF-CONSISTENCY LOOP                        #
#                    (solve for each k-point independently)                   #
#-----------------------------------------------------------------------------#
    Eref = 1.0
    counter=0
    
    Eaux = np.zeros(Kp_x*Kp_y*Ntot)
    
    VHG2    = np.zeros((Mdim2,Mdim2), dtype=complex)
    VHG     = np.zeros((N_G,N_G), dtype=complex)
    VHG_old = np.zeros((N_G,N_G), dtype=complex)
    VXG     = np.zeros((N_G,N_G), dtype=complex)
    VXG_old = np.zeros((N_G,N_G), dtype=complex)
    
    Tmat = np.zeros((N_G*N_G,N_G*N_G), dtype=complex)
    Hmat = np.zeros((N_G*N_G,N_G*N_G), dtype=complex)
    Umat = np.zeros((N_G*N_G,N_G*N_G), dtype=complex)
    
    while abs(Eref-E[1,0,0])>TOL:
        counter+=1
        print("Iteration no.", counter, "criterion = ",abs(Eref-E[1,0,0])) 
        Eref = E[1,0,0]
            
        Umat = Umat_def(N_G,Gnum,A,B,VHG,VXG)
    
        for kxi in range (Kp_x):
            for kyi in range (Kp_y):
                for Gxi in range(N_G):
                    for Gyi in range(N_G):
                        Tmat[Gyi + Gxi*N_G,Gyi + Gxi*N_G] = 0.5*( (kx[kxi]-G[Gxi])**2 + (ky[kyi]-G[Gyi])**2 )
                            
                Hmat = Tmat + Umat 
            
                vals, vecs = np.linalg.eigh(Hmat)
                for l in range(Ntot):
                    E[l,kxi,kyi] = vals[l]
                    Eaux[l + kyi*Ntot + kxi*Kp_y*Ntot]=E[l,kxi,kyi]
                    for m in range (N_G**2):
                        C[l,kxi,kyi,m] = vecs[m,l]  
    
    #   find the Fermi energy and the occupation numbers
        sorted = np.argsort(Eaux)  
        EF_index = Kp_x*Kp_y*Nel//2 - 1
        EF = Eaux[sorted[EF_index]]
        print("Fermi energy:",round(EF,5)) 
        for i in range(Kp_x):
            for j in range(Kp_y):
                for l in range(Ntot):
                    if E[l,i,j]<=EF: 
                        OccNum[l,i,j] = 1
                    else:
                        OccNum[l,i,j] = 0      
                        
    #   find the band gap
        Eg = sys.maxsize
        for kxi in range(Kpoints):
            for kyi in range(Kpoints):
                Omega = E[Nel//2,kxi,kyi] - E[Nel//2-1,kxi,kyi]
                if Omega < Eg:Eg = Omega
        print("Band gap:",round(Eg,5))
                 
        nG = Density(Ksym,OccNum,Ntot,Kp_x,Kp_y,Gnum,dk,C,G,c,HXC,GSout)  
           
        VHG_old = VHG.copy()           
        VHG = VHartree(HXC,Gnum,nG,G2)   
        VHG = MIX*VHG + (1.-MIX)*VHG_old
        
        VXG_old = VXG.copy() 
        VXG = Vexchange(HXC,Gnum,nG,G,G2,c)
        VXG = MIX*VXG + (1.-MIX)*VXG_old
        
        if HXC==0: Eref = E[1,0,0]   # because in this case we want only 1 iteration,
                                     # and this will exit the loop        
#-----------------------------------------------------------------------------#
#                       END OF SELF-CONSISTENCY LOOP                          #
#-----------------------------------------------------------------------------#
    if GSout==1: den_diag_out(c,Mdim2,nG,G2)
    if GSout==2: den_full_out(c,Mdim2,nG,G2)
    if GSout==3: band_out(c,N_G,Umat,G,EF)
    
    return E, C, Eg, nG, OccNum
    
#if Dielectric == 1 or eps_omega_calc==1 or Dipole==1:      
def DipoleMatrixElements(OccNum,C,E):
    #   Calculate the dipole matrix elements ME (along the x-direction).
    #   Notice that we want to avoid matrix elements between near-degenerate
    #   bands. To do this we demand that the energies are separated by more than 0.1,
    #   which happens here (*). The fudge factor 0.1 will need to be explained later.
    
    import numpy as np
    
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

#-----------------------------------------------------------------------------#
#    This function calculates the frequency-dependent dielectric function     #
#-----------------------------------------------------------------------------#
def EpsilonOmega(OccNum,ME,E,Eg):
    
    import numpy as np
    import matplotlib.pyplot as plt
            
    omega = np.zeros(omegavalues)
    epsilon = np.zeros(omegavalues, dtype=complex)
    chi_k = np.zeros((Kp_x,Kp_y), dtype=complex)
    ione = 0. + 1.j   
                          
    for omega_i in range(omegavalues):
        om = omega_i*d_omega
        omega[omega_i] = om
        if omega_i%10==0: print(omega_i,'\tout of',omegavalues)
           
        for kxi in range(Kp_x):
            for kyi in range(Kp_y):
                temp = 0.0
                for j in range(Ntot):#                                                   <- change index name
                    if OccNum[j,kxi,kyi]==1:
                        for l in range(Ntot):#                                           <- change index name
                            if OccNum[l,kxi,kyi]==0:
                                temp += abs(ME[j,l,kxi,kyi])**2  \
                                   *( 1/(E[j,kxi,kyi] - E[l,kxi,kyi] + om + ione*eta)\
                                     +1/(E[j,kxi,kyi] - E[l,kxi,kyi] - om - ione*eta) )
                chi_k[kxi,kyi] = temp
                
        chi = 0.0
        for kxi in range(Kp_x):
            for kyi in range(Kp_y):
                chi += chi_k[kxi,kyi] 
        
        chi = ((2.0/c**2)*chi*(dk**2)) / ((2*np.pi)**2)
#               ^ factor of 2 because of spin
        if Ksym==1:
            chi= 4*chi      
        if Ksym==2 or Ksym==3:
            chi = 2*chi
        
        if quasi2D == 0: epsilon[omega_i] = 1.0 - (2.*np.pi/kval)*(kval**2)*chi
        if quasi2D == 1: epsilon[omega_i] = 1.0 - (4.*np.pi)*chi

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
