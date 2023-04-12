import matplotlib.pyplot as plt
import numpy as np
import math
from   scipy.fft         import fft

E0          =  0.2
alpha_xc    =  -0.25

c           =  5.0

Tsteps      = 5000
dt          = 0.1
TW          = 0
TEND        = Tsteps*dt

n_omega     =  200
d_omega     =  0.01

f = open('dip_t_E0=0.2alpha_xc=0.0.txt','r')

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

lines = f.readlines()
print(lines[0].split()[0])

i = -1
for line in lines:
    i += 1
    if i >= len(lines): exit
    
    Time[i]        = float(line.split()[0])
    eps_t_xc[i]    = float(line.split()[1])

f.close()

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