import numpy as np
import cmath
import matplotlib.pyplot as plt
import math
from scipy.fft import fft
import sys

c = 5.
E0 = 0.01
Tsteps = 1000 
dt = 0.5
TW = 0
TEND = Tsteps*dt
ione = 0. +1.j
Time = np.zeros(Tsteps)
eps = np.zeros(Tsteps)
damp = np.zeros(Tsteps)
window = np.zeros(Tsteps)
integrand = np.zeros(Tsteps, dtype=complex)

for i in range(Tsteps):
    T = i*dt
    Time[i] = T
    damp[i] = math.exp(-T*0.01)
    window[i] = 1.
    if T<TW:
        window[i] = math.sin((T/TW)*math.pi/2.)**0.25
    if T>TEND-TW-dt:
        window[i] = abs(math.sin(((TEND-T-dt)/TW)*math.pi/2.))**0.25
    
#plt.plot(Time,window)
#plt.plot(Time,damp)
#sys.exit()    
   
f = open("j00.txt", "r")
i = -1
for line in f.readlines():
    i += 1
    list1 = line.split()
    Time[i] = float(list1[0])
    eps[i] = float(list1[1]) 
f.close()

#plt.plot(Time,eps*damp)
#sys.exit()

domega = 2*math.pi/TEND
OMM = np.zeros(Tsteps)

for i in range(Tsteps):    
    OMM[i] = i*domega 

y = dt*fft(eps*damp*window)/E0 

y = 1 + ione*y*(4.*math.pi/c**2)/OMM

plt.xlim(0.0,2)
plt.ylim(-0.5,2.5)
plt.plot(OMM,y.real)
plt.plot(OMM,-y.imag)


