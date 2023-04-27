E0          =  0.2
alpha_xc    =  -0.25

c           =  5.0

Tsteps      =  1000
dt          =  0.5

n_omega     =  200
d_omega     =  0.01

#*****************************************************************************#
#****************************USER INPUT ENDS HERE*****************************#
#*****************************************************************************#

import numpy             as np
import matplotlib.pyplot as plt
import math
from   scipy.fft         import fft
import os

location    = "./Results/E0=" + str(E0) + "alpha_xc=" + str(alpha_xc)
datatype    = "dip_t"
parameters  = "_E0=" + str(E0) + "alpha_xc=" + str(alpha_xc)
extension   = ".txt"
filename    = datatype + parameters + extension
path        = os.path.join(location, filename)

TW          = 0
TEND        = Tsteps*dt

Time        = np.zeros(Tsteps)
eps_t       = np.zeros(Tsteps)
eps_t_xc    = np.zeros(Tsteps)

Omega       = np.zeros(n_omega)
eps_w       = np.zeros(n_omega,  dtype=complex)

damp        = np.zeros(Tsteps)
window      = np.zeros(Tsteps)
integrand   = np.zeros(Tsteps, dtype=complex)

for i in range(Tsteps):
    T = i*dt
    Time[i] = T
    damp[i] = math.exp(-T*0.01)
    window[i] = 1.0
    if T<TW:
        window[i] = math.sin((T/TW)*math.pi/2.0)**0.25
    if T>TEND-TW:
        window[i] = abs(math.sin(((TEND-T-dt)/TW)*math.pi/2))**0.25

# read in the frequency-dependent dielectric function data
f = open('./Results/eps(w).txt', 'r')

i =-1
for line in f.readlines():
    i += 1
    
    list2    = line.split()
    Omega[i] = float(list2[0])
    eps_r    = float(list2[1])
    eps_i    = float(list2[2])
    eps_w[i] = eps_r + 1j*eps_i

f.close()

# read in the time-dependent dipole moment data    
f = open(path, 'r')

i = -1
for line in f.readlines():
    i += 1
    
    list1     = line.split()
    Time[i]   = float(list1[0])
    eps_t[i]  = float(list1[1])  

f.close()

domega = 2*math.pi/TEND
OMM    = np.zeros(Tsteps)

for i in range(Tsteps):    
    OMM[i] = i*domega 

y = dt*fft(eps_t*damp*window)/E0

y = 1.0 - y*4*math.pi/c**2

print(y[0])

plt.title("Time-Dependent Dipole Moment\n with E0 = " + str(E0) + " and alpha_xc = " + str(alpha_xc))
plt.xlabel("Time")
plt.ylabel("Dipole Moment")
plt.xlim( 0.0,100.0)
plt.plot(Time, eps_t, color='blue')
plt.axhline(y=0, color='black', ls='--')
plt.show()

plt.title("Fourier Transform of the Time-Dependent Dielectric Function\n with E0 = " + str(E0) + " and alpha_xc = " + str(alpha_xc))
plt.xlabel("Frequency (Energy)")
plt.ylabel("Dielecrtic Function")
plt.xlim( 0.0,1.0)
plt.ylim(-0.1,2.5)
plt.plot(OMM,   y.real,     label="FT - Real",      color='blue')
plt.plot(Omega, eps_w.real, label="LR - Real",      color='red')
plt.plot(OMM,   -y.imag,    label="FT - Imaginary", color='blue', ls='--')
plt.plot(Omega, eps_w.imag, label="LR - Imaginary", color='red',  ls='--')
plt.legend()
plt.show()

# read in the time-dependent excited state population with alpha_xc = 0.0,-0.25
Tsteps      =  10000
dt          =  0.01

Time        = np.zeros(Tsteps)
eps_t       = np.zeros(Tsteps)
eps_t_xc    = np.zeros(Tsteps)

Omega       = np.zeros(n_omega)
eps_w       = np.zeros(n_omega,  dtype=complex)

damp        = np.zeros(Tsteps)
window      = np.zeros(Tsteps)
integrand   = np.zeros(Tsteps, dtype=complex)

location    = "./Results/E0=" + str(E0) + "alpha_xc=0.0"
datatype    = "dip_t"
parameters  = "_E0=" + str(E0) + "alpha_xc=0.0"
extension   = ".txt"
filename    = datatype + parameters + extension
path        = os.path.join(location, filename)
f_Nex = open(path,'r')

i = -1
for line in f_Nex.readlines():
    i += 1
    
    list1     = line.split()
    Time[i]   = float(list1[0])
    eps_t[i]  = float(list1[1])   

f_Nex.close()

domega = 2*math.pi/TEND
OMM    = np.zeros(Tsteps)

for i in range(Tsteps):    
    OMM[i] = i*domega 

y = dt*fft(eps_t*damp*window)/E0

y = 1.0 - y*4*math.pi/c**2

plt.plot(OMM,   y.real,     label="No XC - Real",      color='red')
plt.plot(OMM,   -y.imag,    label="No XC - Imaginary", color='red', ls='--')

Tsteps      =  1000
dt          =  0.5

Time        = np.zeros(Tsteps)
eps_t       = np.zeros(Tsteps)
eps_t_xc    = np.zeros(Tsteps)

Omega       = np.zeros(n_omega)
eps_w       = np.zeros(n_omega,  dtype=complex)

damp        = np.zeros(Tsteps)
window      = np.zeros(Tsteps)
integrand   = np.zeros(Tsteps, dtype=complex)

location    = "./Results/E0=" + str(E0) + "alpha_xc=-0.25"
datatype    = "dip_t"
parameters  = "_E0=" + str(E0) + "alpha_xc=-0.25"
extension   = ".txt"
filename    = datatype + parameters + extension
path        = os.path.join(location, filename)
f_NexAxc = open(path,'r')

i = -1
for line in f.readlines():
    i += 1
    
    list1        = line.split()
    Time[i]      = float(list1[0])
    eps_t_xc[i]  = float(list1[1])

f_NexAxc.close()

domega = 2*math.pi/TEND
OMM    = np.zeros(Tsteps)

for i in range(Tsteps):    
    OMM[i] = i*domega 

y = dt*fft(eps_t*damp*window)/E0

y = 1.0 - y*4*math.pi/c**2

plt.title("Fourier Transform of the Time-Dependent Dielectric Function\n with E0 = " + str(E0) + " and alpha_xc = " + str(alpha_xc))
plt.xlabel("Frequency (Energy)")
plt.ylabel("Dielecrtic Function")
plt.xlim( 0.0,1.0)
plt.ylim(-0.1,2.5)
plt.plot(OMM,   y.real,     label="XC - Real",      color='blue')
plt.plot(OMM,   -y.imag,    label="XC - Imaginary", color='blue', ls='--')
plt.legend()
plt.show()
