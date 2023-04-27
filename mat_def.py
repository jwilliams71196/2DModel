import numpy as np
import math
#
#------------------------------------------------------------------------------
#    Umat_def defines the scalar potential
#------------------------------------------------------------------------------
def Umat_def(Mdim,Gnum,A,B,VHG,VXG):
    U = np.zeros((Mdim,Mdim), dtype=complex)
    Umat = np.zeros((Mdim*Mdim,Mdim*Mdim), dtype=complex)
#
    for i in range (Mdim):
        jx = i-Gnum
        for j in range (Mdim):
            jy = j - Gnum
            if jx==0 and abs(jy)==1:
                U[i,j] = -(A-B)/2.
            if jy==0 and abs(jx)==1:
                U[i,j] = -(A-B)/2.
            if abs(jx)==1 and abs(jy)==1:
                U[i,j] = -(A+B)/4.
                
    U += VHG 
    U += VXG           
#
    for i in range (Mdim):
        jx = i-Gnum
        for j in range (Mdim):
            jy = j-Gnum
            for ip in range (Mdim):
                jxp = ip-Gnum
                for jp in range (Mdim):
                    jyp = jp-Gnum
                
                    djx = jxp - jx
                    djy = jyp - jy
                
                    if (abs(djx)<=Gnum and abs(djy)<=Gnum):
                        Umat[i + j*Mdim,ip + jp*Mdim] = U[djx+Gnum,djy+Gnum] 
    return Umat
#------------------------------------------------------------------------------
#   Amat_def defines the time-dependent vector potential
#------------------------------------------------------------------------------
def Amat_def(Mdim,Gnum,C):
    AVx = np.zeros((Mdim,Mdim), dtype=complex)
    AVy = np.zeros((Mdim,Mdim), dtype=complex)
    Axmat = np.zeros((Mdim*Mdim,Mdim*Mdim), dtype=complex)
    Aymat = np.zeros((Mdim*Mdim,Mdim*Mdim), dtype=complex)
    A2mat = np.zeros((Mdim*Mdim,Mdim*Mdim), dtype=complex)
    ione = 0. + 1.j
#
    for i in range (Mdim):
        jx = i-Gnum
        for j in range (Mdim):
            jy = j - Gnum
            if jx==0 and abs(jy)==1:
                AVy[i,j] = math.pi*C*ione*jy
            if jy==0 and abs(jx)==1:
                AVx[i,j] = math.pi*C*ione*jx
            if abs(jx)==1 and abs(jy)==1:
                AVx[i,j] = 0.5*math.pi*C*ione*jx
                AVy[i,j] = 0.5*math.pi*C*ione*jy
#
    for i in range (Mdim):
        jx = i-Gnum
        for j in range (Mdim):
            jy = j-Gnum
            for ip in range (Mdim):
                jxp = ip-Gnum
                for jp in range (Mdim):
                    jyp = jp-Gnum
                
                    djx = jxp - jx
                    djy = jyp - jy
                
                    if (abs(djx)<=Gnum and abs(djy)<=Gnum):
                        Axmat[i + j*Mdim,ip + jp*Mdim] = AVx[djx+Gnum,djy+Gnum]
                        Aymat[i + j*Mdim,ip + jp*Mdim] = AVy[djx+Gnum,djy+Gnum]     

                    dum = 0.
                    for ipp in range (Mdim):
                        jxpp = ipp-Gnum
                        for jpp in range (Mdim):
                            jypp = jpp - Gnum
                            
                            d1jx = jxp - jxpp
                            d2jx = jxpp - jx
                            d1jy = jyp - jypp
                            d2jy = jypp - jy
                            if (abs(d1jx)<=Gnum and abs(d1jy)<=Gnum \
                                and abs(d2jx)<=Gnum and abs(d2jy)<=Gnum):
                                    dum += \
                           AVx[d1jx+Gnum,d1jy+Gnum]*AVx[d2jx+Gnum,d2jy+Gnum] +\
                           AVy[d1jx+Gnum,d1jy+Gnum]*AVy[d2jx+Gnum,d2jy+Gnum]
                       
                    A2mat[i + j*Mdim,ip + jp*Mdim] = 0.5*dum
                                                               
    return Axmat,Aymat,A2mat