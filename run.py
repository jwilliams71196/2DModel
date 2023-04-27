import time
start_time = time.time()
#-----------------------------------------------------------------------------#

import subprocess, sys, os

import numpy as np

#-----------------------------------------------------------------------------#
#---------------------------- INITIALIZATIONS --------------------------------#
#-----------------------------------------------------------------------------#

from infile import *
from initializations import *

#-----------------------------------------------------------------------------#
#------------------------- GROUND STATE CALCULATION --------------------------#
#-----------------------------------------------------------------------------#

from groundstate import *

E,C,Eg,nG, OccNum = GroundState()

if Dielectric == 1 or eps_omega_calc==1 or Dipole==1: ME = DipoleMatrixElements(OccNum,C,E)

#-----------------------------------------------------------------------------#
#------------------------------ EPSILON(OMEGA) -------------------------------#
#-----------------------------------------------------------------------------#

if eps_omega_calc==1:
    
    EpsilonOmega(OccNum,ME,E,Eg)
    
#-----------------------------------------------------------------------------#
#-------------------------------- EPSILON(t) ---------------------------------#
#-----------------------------------------------------------------------------#

if timeprop == 0: 
    end_time = time.time()
    print("Completed in " + str(end_time - start_time) + "s")
    sys.exit()

#-----------------------------------------------------------------------------#
#----------------------------- POST PROCESSING -------------------------------#
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
end_time = time.time()
print("Completed in " + str(end_time - start_time) + "s")