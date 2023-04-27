import math
import cmath
import numpy as np
import matplotlib.pyplot as plt
#

def den_diag_out(c,Mdim2,nG,G2x,G2y):
#
#  Calculate and plot the real space density along the diagonal
#             
    ione = 0. + 1.j   
    Xpoints = 41
    x = np.zeros(Xpoints)
    den = np.zeros(Xpoints)
    dx = c/(Xpoints-1)
    for ii in range(Xpoints):
        xx = -c/2 + ii*dx
        yy = xx
        x[ii] = xx
    
        dum = 0.
        for i in range(Mdim2):
            for j in range(Mdim2):
                dum += nG[i,j]*cmath.exp(ione*(G2x[i]*xx+G2y[j]*yy))
        den[ii] = dum.real
    plt.plot(x,den)
    plt.show()  
#---------------------------------------------------------------------------    
def den_full_out(c,Mdim2,nG,G2x,G2y):
#
#  Calculate and plot the 2D real space density
#   
    ione = 0. + 1.j             
    Xpoints = 31
    x = np.zeros(Xpoints)
    den = np.zeros((Xpoints,Xpoints))
    dx = c/(Xpoints-1)
    for ii in range(Xpoints):
        xx = -c/2 + ii*dx
        x[ii] = xx
        for jj in range(Xpoints):        
            yy = -c/2 + jj*dx
            dum = 0.
            for i in range(Mdim2):
                for j in range(Mdim2):
                    dum += nG[i,j]*cmath.exp(ione*(G2x[i]*xx+G2y[j]*yy))
            den[ii,jj] = dum.real
          
    y = x.copy()
    X,Y = np.meshgrid(x,y)
    fig = plt.figure(figsize =(14, 9))
    ax = plt.axes(projection ='3d')
    ax.plot_surface(X, Y, den)
    plt.show()    
#---------------------------------------------------------------------------    
def band_out(c,Mdim,Umat,G,EF):

#  Calculate band structure along Gamma --> X --> M --> Gamma in the BZ

    Nbands = 5
    Kpoints = 51
    EB = np.zeros((Nbands,3*Kpoints-2))
    dk = (math.pi/c)/(Kpoints-1)
    kx = np.zeros((3*Kpoints-2))
    ky = np.zeros((3*Kpoints-2))
    Tmat = np.zeros((Mdim*Mdim,Mdim*Mdim), dtype=complex)
    for i in range (Kpoints):
        kx[i] = i*dk
        ky[i] = 0.
    for i in range (Kpoints-1):
        kx[i+Kpoints] = math.pi/c
        ky[i+Kpoints] = (i+1)*dk
    for i in range (Kpoints-1):
        kx[i+2*Kpoints-1] = math.pi/c - (i+1)*dk
        ky[i+2*Kpoints-1] = math.pi/c - (i+1)*dk

    for i in range (3*Kpoints-2):   
        for i1 in range(Mdim):
            for i2 in range(Mdim):
                Tmat[i2 + i1*Mdim,i2 + i1*Mdim] = \
                    0.5*( (kx[i]-G[i1])**2 + (ky[i]-G[i2])**2 )
            
        Hmat = Tmat + Umat               
        vals, vecs = np.linalg.eigh(Hmat)  
        for j in range(Nbands):
            EB[j,i] = vals[j]
            
    EB = EB-EF        
    
    plt.plot(EB[0])
    plt.plot(EB[1])
    plt.plot(EB[2])
    plt.plot(EB[3])
    plt.plot(EB[4])
    
    plt.xlabel("k")
    plt.ylabel("E")
    plt.title('$\Gamma$                         X                          M                         $\Gamma$') # lol
    plt.tick_params(labelbottom = False, bottom = False)

#find the minimum and maximum of each band

    Emm = np.zeros((10))
    kmm = np.zeros((10))
    for i in range(5):
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
    print("min/max of Band 2:", Emm[2], Emm[3])
    print("min/max of Band 3:", Emm[4], Emm[5])
    print("min/max of Band 4:", Emm[6], Emm[7])
    plt.plot(kmm,Emm,'o')    
    Eout = np.zeros(2)
    EFout = np.zeros(2)
#    EFout[0]=EF
#    EFout[1]=EF
    Eout[0]=0
    Eout[1]=150
    plt.plot(Eout,EFout,linestyle='dashed',color='black')
    plt.show()