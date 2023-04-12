import numpy as np
#-----------------------------------------------------------------------------#
#    This subroutine calculates the time-dependent dielectric function        #
#-----------------------------------------------------------------------------#
def Dielectric_t(Ksym,OccNum,Ntot,Kp_x,Kp_y,Mdim,dk,kval,Ct,C,ME,E,c,quasi2D):

    beta = np.zeros((Ntot,Ntot,Kp_x,Kp_y), dtype=complex)
    ck = np.zeros((Kp_x,Kp_y))

    for i in range (Kp_x):
        for j in range (Kp_y):
            for l in range (Ntot):
                if OccNum[l,i,j]==1:
                    for m in range(Ntot):
                        temp = 0.
                        for k in range (Mdim*Mdim):
                            temp += Ct[l,i,j,k]*np.conj(C[m,i,j,k])
                        beta[l,m,i,j] = temp

    for i in range(Kp_x):
        for j in range(Kp_y):
            CC = 0.
            for j1 in range(Ntot):
                if OccNum[j1,i,j]==1:
                    for k in range(Ntot):
                        for l in range(Ntot):
                            temp = 0.
                            for m in range(Ntot):
                                if m != k:
                                    temp += ME[l,m,i,j]*ME[m,k,i,j]/(E[k,i,j] - E[m,i,j])
                            CC += temp*beta[j1,k,i,j]*np.conj(beta[j1,l,i,j])
            ck[i,j] = 2.*CC.real

#   Now integrate over the Brillouin zone
    temp= 0.0
    for i in range(Kp_x):
        for j in range(Kp_y):
            temp += ck[i,j]

    CHI = -(2./c**2)*temp*dk**2/(2*np.pi)**2    # factor 2 because of spin
    if Ksym==1:
        CHI = 4*CHI
    if Ksym==2 or Ksym==3:
        CHI = 2*CHI

    if quasi2D == 0: EPS = 1. - (2.*np.pi/kval)*kval**2*CHI
    if quasi2D == 1: EPS = 1. - 4.*np.pi*CHI

    return EPS

#------------------------------------------------------------------------------
#    This subroutine calculates the time-dependent dipole moment
#------------------------------------------------------------------------------
def Dipole_t(Ksym,OccNum,Ntot,Kp_x,Kp_y,Mdim,dk,Ct,C,ME,E):

    xi = np.zeros((Ntot,Ntot,Kp_x,Kp_y), dtype=complex)
    ck = np.zeros((Kp_x,Kp_y))

    for k1 in range (Kp_x):
        for k2 in range (Kp_y):
            for i in range (Ntot):
                if OccNum[i,k1,k2]==1:
                    for m in range(Ntot):
                        temp = 0.0
                        for j in range (Mdim*Mdim):
                            temp += Ct[i,k1,k2,j]*np.conj(C[m,k1,k2,j])
                        xi[i,m,k1,k2] = temp

    for k1 in range(Kp_x):
        for k2 in range(Kp_y):
            CC = 0.0
            for i in range(Ntot):
                if OccNum[i,k1,k2]==1:
                    for n in range(Ntot):
                        for m in range(Ntot):
                            if m != n:
                               CC += np.conj(xi[i,n,k1,k2])*xi[i,m,k1,k2]*ME[n,m,k1,k2]
            ck[k1,k2] = CC.imag

#   Now integrate over the Brillouin zone
    temp= 0.0
    for i in range(Kp_x):
        for j in range(Kp_y):
            temp += ck[i,j]

    Dip = 2.*temp*dk**2/(2*np.pi)**2    # factor 2 because of spin
    if Ksym==2 or Ksym==3:
        Dip = 2*Dip

    return Dip
