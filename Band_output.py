import math
import cmath
import numpy             as     np
import matplotlib.pyplot as     plt
from   infile            import c,Ntot,Gnum,Kpoints,k_choice,Nel,A,B

# Define the global prameters for these functions
N_G        = 2*Gnum+1
N_Gpts     = 2*N_G-1
dk         = (2 * np.pi) / (Kpoints * c)
G0         = (2 * np.pi) / c
q          = np.sqrt( k_choice[0]**2 + k_choice[1]**2 ) * dk
k          = np.zeros(Kpoints)
G          = np.zeros(N_G)
Gpts       = np.zeros(N_Gpts)
for ki in range (Kpoints): k[ki]    = (ki+0.5)*dk - np.pi/c
for Gi in range (N_G):     G[Gi]    = G0*(Gi-Gnum)
for Gi in range (N_Gpts):  Gpts[Gi] = G0*(Gi-2*Gnum)

#%%---------------------------------------------------------------------------#
#
#-----------------------------------------------------------------------------#
def den_diag_out(nG):

    # Calculate and plot the real space density along the diagonal
    Xpoints = 41
    x   = np.zeros(Xpoints)
    den = np.zeros(Xpoints)
    dx  = c/(Xpoints-1)
    
    for xi in range(Xpoints):
        x[xi] = -c/2 + xi*dx
        yy = 0 # ?
    
        density = 0
        for i in range(N_Gpts):
            for j in range(N_Gpts):
                density += nG[i,j]*cmath.exp(1j*(Gpts[i]*x[xi]+Gpts[j]*yy))
        den[xi] = density.real
    
    plt.title('Electron Density along the Diagonal of the Brillouin Zone')
    plt.xlabel('k')
    plt.ylabel('electron density')
    plt.plot(x,den)
    plt.show()  
#-----------------------------------------------------------------------------#
# 
#-----------------------------------------------------------------------------#
def den_full_out(nG):

    # Calculate and plot the 2D real space density   
    Xpoints = 31
    x, y = np.zeros(Xpoints), np.zeros(Xpoints)
    den = np.zeros((Xpoints,Xpoints))
    dx = c/(Xpoints-1)
    
    for xi in range(Xpoints):
        x[xi] = -c/2 + xi*dx
        for yi in range(Xpoints):        
            y[yi] = -c/2 + yi*dx
            density = 0
            for Gxi in range(N_Gpts):
                for Gyi in range(N_Gpts):
                    density += nG[Gxi,Gyi]                                         \
                           * cmath.exp(1j*(Gpts[Gxi]*x[xi] + Gpts[Gyi]*y[yi]))
            den[xi,yi] = density.real
     
    plt.title('Full Electron Density in the Brillouin Zone')
    plt.xlabel('$k_x$')
    plt.ylabel('$k_y$')
    plt.zlabel('electron density')
    
    y    = x.copy()
    X, Y = np.meshgrid(x,y)
    
    plt.figure(figsize =(14, 9))
    ax = plt.axes(projection ='3d')
    ax.plot_surface(X, Y, den)
    plt.show()    
#---------------------------------------------------------------------------    
def band_out(Umat,EF):
#
#  Calculate band structure along Gamma --> X --> M --> Gamma in the BZ
#
    Kpoints_out = 200
    dk_out = (np.pi / c) / (Kpoints_out - 1)
    kx_out = np.zeros((3*Kpoints_out-2))
    ky_out = np.zeros((3*Kpoints_out-2))
    Tmat   = np.zeros((N_G**2,N_G**2), dtype=complex) 
    EB     = np.zeros((Ntot,3*Kpoints_out-2))

    
    for i in range (Kpoints_out):
        kx_out[i] = i*dk_out
        ky_out[i] = 0
    for i in range (Kpoints_out-1):
        kx_out[i+Kpoints_out] = np.pi / c
        ky_out[i+Kpoints_out] = (i+1)*dk_out
    for i in range (Kpoints_out-1):
        kx_out[i+2*Kpoints_out-1] = np.pi / c - (i+1)*dk_out
        ky_out[i+2*Kpoints_out-1] = np.pi / c - (i+1)*dk_out

    kPathLength  = 3*Kpoints_out-2
    kPathPercent = kPathLength // 10 + 1
    percentDone  = 0

    print()
    for i in range (3*Kpoints_out-2):
        if   (i == 0): 
            print("calculating band structure for",3*Kpoints_out-2,            \
                  "k-points: [", end="")
        elif ((i%kPathPercent) == 0):
            percentDone += 10
            print(str(percentDone) + "%-", end='')

        for i1 in range(N_G):
            for i2 in range(N_G):
                Tmat[i2 + i1*N_G,i2 + i1*N_G] =                              \
                    0.5*( (kx_out[i]-G[i1])**2 + (ky_out[i]-G[i2])**2 )
                    
        Hmat = Tmat + Umat
            
        vals, vecs = np.linalg.eigh(Hmat)        
        for j in range(Ntot):
            EB[j,i] = vals[j]

    print("100%]")

    EB = EB-EF        
    
    for i in range(Ntot):
        if (i < Nel/2): 
            plt.plot(EB[i],label=str(i+1), color='red',  linewidth=0.85)
        else: 
            plt.plot(EB[i],label=str(i+1), color='blue', linewidth=0.85)
    plt.xlim([0,3*Kpoints_out-3])
    plt.title('Electronic Band Structure for A = ' \
              + str(A) + '\tB = ' + str(B) )
    plt.xlabel("$\Gamma$\t\t\t\t\t X\t\t\t\t\t M\t\t\t\t\t $\Gamma$\nk")
    plt.ylabel("E - $E_F$")
    plt.tick_params(labelbottom = False, bottom = False)
    plt.axvline(Kpoints_out,     color='black')
    plt.axvline(2*Kpoints_out+1, color='black')

    # find the minimum and maximum of each band
    Emm = np.zeros((2*Ntot))
    kmm = np.zeros((2*Ntot))
    for i in range(int(Ntot)):
        m1 = EB[i,0]
        m2 = EB[i,0]
        km1=0
        km2=0
        for j in range (3*Kpoints_out-2):
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
    for i in range(0,2*Ntot-1,2):
        print("min/max of Band",((i//2)+1),":",Emm[i],Emm[i+1])
        if (np.ceil((i/2)) == ((Nel/2)-1)): 
            print("------------------------------------------")

#    plt.plot(kmm,Emm,'o')    
    Eout  = np.zeros(2)
    EFout = np.zeros(2)
#    EFout[0]=EF
#    EFout[1]=EF
    Eout[0]=0
    Eout[1]=3*Kpoints_out-3
#    plt.plot(Eout,EFout,linestyle='dashed',color='black')
    plt.show()