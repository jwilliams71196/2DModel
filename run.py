import time
start_time = time.time()
# ----------------------------------------------------------------------------#

import sys

import numpy as np

# ----------------------------------------------------------------------------#
# --------------------------- INITIALIZATIONS --------------------------------#
# ----------------------------------------------------------------------------#

from infile import *
from initializations import *

# ----------------------------------------------------------------------------#
# ------------------------ GROUND STATE CALCULATION --------------------------#
# ----------------------------------------------------------------------------#

from groundstate import *

E,C,Eg,nG, OccNum = GroundState()

if Dielectric == 1 or eps_omega_calc==1 or Dipole==1: ME = DipoleMatrixElements(OccNum,C,E)

# ----------------------------------------------------------------------------#
# ----------------------------- EPSILON(OMEGA) -------------------------------#
# ----------------------------------------------------------------------------#

if eps_omega_calc==1:
    EpsilonOmega(OccNum,ME,E,Eg)

# ----------------------------------------------------------------------------#
# ------------------------------- EPSILON(t) ---------------------------------#
# ----------------------------------------------------------------------------#

if timeprop == 0: 
    end_time = time.time()
    elapsed_time = float(end_time) - float(start_time)
    print("Completed in", round(elapsed_time, 5), "s")
    sys.exit()

if Perturbation > 1:
    if (theta // 90) != 0:
        if Ksym != 4:
            print('Error: Ksysm must be 4 for a vector potential at an angle to both axes.')
            end_time = time.time()
            print("Exited after " + str(end_time - start_time) + "s")
            sys.exit()
    else:
        if theta//180 == 0:
            if Ksym//2 != 0:
                print('Error: Ksym must be 2 or 4 for a vector potential along the x direction.')
                end_time = time.time()
                print("Exited after " + str(end_time - start_time) + "s")
                sys.exit()
        else:
            if Ksym < 3:
                print('Error: Ksym must be 3 o4 4 for a vector potential along the y direction.')
                end_time = time.time()
                print("Exited after " + str(end_time - start_time) + "s")
                sys.exit()

# ----------------------------------------------------------------------------#
# ---------------------------- POST PROCESSING -------------------------------#
# ----------------------------------------------------------------------------#

# ----------------------------------------------------------------------------#
end_time = time.time()
elapsed_time = float(end_time) - float(start_time)
print("Completed in", round(elapsed_time, 5), "s")
