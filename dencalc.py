import cmath
import math
import numpy as np
import sys
#
#------------------------------------------------------------------------------
#    This subroutine calculates the density, nG, in G-space
#    using occupation numbers and simple integration
#------------------------------------------------------------------------------
def Density(Ksym,OccNum,Ntot,Kp_x,Kp_y,Gnum,dk,C,G,c,HXC,out):
    Mdim = 2*Gnum+1
    Mdim2 = 2*Mdim-1
    nG = np.zeros((Mdim2,Mdim2), dtype=complex)
    nG1 = np.zeros((Mdim2,Mdim2), dtype=complex)
    nGk = np.zeros((2*Mdim,2*Mdim,Kp_x,Kp_y), dtype=complex)  
    PI = math.pi
    
    if HXC==1 or out==1 or out==2:
    
        for i in range(Kp_x):
            for j in range(Kp_y):
                for l in range(Ntot):
                    if OccNum[l,i,j]==1:
                        for i1 in range(Mdim):
                            for j1 in range(Mdim): 
                                m1 = j1 + i1*Mdim                      
                                for i2 in range(Mdim):
                                    for j2 in range(Mdim):
                                        m2 = j2 + i2*Mdim
                                
                                        i3 = i2-i1+2*Gnum
                                        j3 = j2-j1+2*Gnum                             

                                        nGk[i3,j3,i,j] += \
                                    C[l,i,j,m1]*np.conj(C[l,i,j,m2])
                    
#
#   Now integrate over the Brillouin zone
#
        for i3 in range(Mdim2):
            for j3 in range(Mdim2):

                dum = 0.
                for i in range(Kp_x):
                    for j in range(Kp_y):
                        dum += nGk[i3,j3,i,j]
    
                nG1[i3,j3] = 2.*dum*dk**2/(2*PI)**2       
            
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
def current1(Ksym,OccNum,Ntot,Kp_x,Kp_y,Mdim,dk,Ct,Gx,Gy,kx,ky):
    
    jxk = np.zeros((Kp_x,Kp_y))
#    jyk = np.zeros((Kp_x,Kp_y))
    
    for i in range (Kp_x):
        for j in range (Kp_y):
            dumx = 0.
#            dumy = 0. 
            for l in range (Ntot):   
                if OccNum[l,i,j]==1:
                    for i1 in range(Mdim):
                        for i2 in range(Mdim):
                            m = i2 + i1*Mdim
#                            dumx += (Gx[i1]-kx[i])*np.abs(Ct[l,i,j,m])**2
                            dumx += Gx[i1]*np.abs(Ct[l,i,j,m])**2
#                            dumy += Gy[i2]*np.abs(Ct[l,i,j,m])**2
                jxk[i,j] = dumx
#                jyk[i,j] = dumy
#
#   Now integrate over the Brillouin zone
#
    dumx = 0.
#    dumy = 0.
    for i in range(Kp_x):
        for j in range(Kp_y):
            dumx += jxk[i,j]
#            dumy += jyk[i,j]
    jx = 2*dumx*dk**2/(2*math.pi)**2
#    jy = 2*dumy*dk**2/(2*math.pi)**2
    jy = 0.
 
    if Ksym==2 or Ksym==3:
        jx = 2*jx
#        jy = 2*jy
        
    
    return jx,jy
#------------------------------------------------------------------------------
#    This subroutine calculates the number of excited electrons
#------------------------------------------------------------------------------
def excited_pop1(Ksym,OccNum,Ntot,Kp_x,Kp_y,Mdim,dk,C,Ct,c):
  
    nk = np.zeros((Kp_x,Kp_y))   
            
    for i in range (Kp_x):
        for j in range (Kp_y):
            dum1 = 0.
            for l in range (Ntot):
                if OccNum[l,i,j]==1:
                    for m in range(Ntot):
                        if OccNum[m,i,j]==0:                          
                            dum = 0.
                            for k in range (Mdim*Mdim):    
                                dum += Ct[l,i,j,k]*np.conj(C[m,i,j,k])
                            dum1 +=  np.abs(dum)**2
            nk[i,j] = dum1   
    #
    #   Now integrate over the Brillouin zone
    #
    dum = 0.
    for i in range(Kp_x):
        for j in range(Kp_y):
            dum += nk[i,j]
        
    N_ex = 2.*dum*dk**2/(2*math.pi)**2    # factor 2 because of spin
    
    if Ksym==1:
        N_ex= 4*N_ex     
    if Ksym==2 or Ksym==3:
        N_ex = 2*N_ex
        
    return N_ex*c**2
#------------------------------------------------------------------------------
#    This subroutine calculates the occupation numbers as a function
#    of energy.
#------------------------------------------------------------------------------
def Fermi_dist(Ksym,OccNum,Ntot,Kp_x,Kp_y,Mdim,dk,C,Ct):
  
#    Faux = np.zeros((Ntot,Kp_x,Kp_y)) 
    F_occ = np.zeros(Kp_x*Kp_y*Ntot)
            
    for i in range (Kp_x):
        for j in range (Kp_y):
            for l in range (Ntot):
                dum1=0.
                for m in range(Ntot):
                    if OccNum[m,i,j]==1:
                        dum = 0.
                        for k in range (Mdim*Mdim):    
                            dum += Ct[m,i,j,k]*np.conj(C[l,i,j,k])
                        dum1 +=  np.abs(dum)**2
#                Faux[l,i,j] = dum1   
                F_occ[l + j*Ntot + i*Kp_y*Ntot]=dum1
        
    return F_occ