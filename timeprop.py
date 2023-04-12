def TimePropagation():
    
    from mat_def     import Umat_def
    from dencalc     import Density,excited_pop1,current1,Fermi_dist
    from VHX_find    import Vexchange,VHartree
    from Band_output import den_diag_out,den_full_out,band_out
    from epsiloncalc import epsilon_t,epsilon_om,dipole_t
    
    import numpy             as np
    import matplotlib.pyplot as plt
    
    from infile import *
    from initializations import *
#-----------------------------------------------------------------------------#
#                           BEGIN TIME PROPAGATION                            #
#-----------------------------------------------------------------------------#

    for i in range(Mdim**2):
        Identity[i,i]=1.0
        
    VHG2    = np.zeros((Mdim2,Mdim2), dtype=complex)
    VHG     = np.zeros((N_G,N_G), dtype=complex)
    VHG_old = np.zeros((N_G,N_G), dtype=complex)
    VXG     = np.zeros((N_G,N_G), dtype=complex)
    VXG_old = np.zeros((N_G,N_G), dtype=complex)

    Ct = np.copy(C)
    At = A
    Ax = 0.
    Ay = 0.
    Axcx = 0.
    Axcy = 0.

    T = 0
    counter = -1
    while counter < Tsteps-1:
        counter+=1
        T = T + dt
        Time[counter] = T
        T2 = T-dt/2.     # evaluate perturbation at mid-timestep
        if (counter+1)%10==0:
            print()
            print('time =',round(T,6))

        if counter>1:
            Axc_x[counter] = alpha_xc*dt**2*jx_mac[counter-1] \
                + 2*Axc_x[counter-1] - Axc_x[counter-2]
            Axc_y[counter] = alpha_xc*dt**2*jy_mac[counter-1] \
                + 2*Axc_y[counter-1] - Axc_y[counter-2]

            Axcx = Axc_x[counter]
            Axcy = Axc_y[counter]

    #   Envelope of the time-dependent perturbation
        if Perturbation==1 or Perturbation==2:
            if T < Ncycles*2*PI/omega_dr:
                ENV = 1.
            else:
                ENV = 0.

    #   SCALAR POTENTIAL:
        if Perturbation==1: At =  A*(1. + alpha_t*math.sin(omega_dr*T2)*ENV)

    #  VECTOR POTENTIAL (x and y components):

    #   a pulsed scalar potential
        if Perturbation==2:
            AA = (E0/omega_dr)*math.cos(omega_dr*T2)*ENV
            Ax = AA*math.cos(theta*PI/180.)
            Ay = AA*math.sin(theta*PI/180.)

    #   a pulsed vector potential
        if Perturbation==3:
            Ax = E0
            Ay = 0.0

    #   switch on a constant electric field
        if Perturbation==4:
            Ax = E0*T2
            Ay = 0.

    #   now starts the predictor-corrector loop
        pc_count=-1
        while pc_count < Ncorr:
            pc_count += 1

            if pc_count == 0:
                nG0 = densityG1(Ksym,OccNum,Ntot,Kp_x,Kp_y,Gnum,dk,Ct,Gx,Gy,HXC,out)
                nG = np.copy(nG0)

            VHG = VHartree(HXC,Gnum,nG,G2x,G2y)
            VXG = Vexchange(HXC,Gnum,nG,Gx,Gy,G2x,G2y,c)

            Umat = Umat_def(Mdim,Gnum,At,B,VHG,VXG)

            for i in range (Kp_x):
                for j in range (Kp_y):
                    for i1 in range(Mdim):
                        for i2 in range(Mdim):
                            Tmat[i2 + i1*Mdim,i2 + i1*Mdim] = \
                             0.5*( (kx[i]-Gx[i1]+Ax+Axcx)**2 + (ky[j]-Gy[i2]+Ay+Axcy)**2 )
                    Hmat = Tmat + Umat

                    for l in range(Ntot):
                        if OccNum[l,i,j]==1:
                            for m in range(Mdim*Mdim):
                                PHI[m] = Ct[l,i,j,m]

                            Mmat = Identity - 0.5*dt*ione*Hmat
                            RHS = Mmat.dot(PHI)
                            Mmat = Identity + 0.5*dt*ione*Hmat
                            PHI = np.linalg.solve(Mmat,RHS)

                            for m in range(Mdim*Mdim):
                                Ct1[l,i,j,m] = PHI[m]

            if Ncorr>0:
                nG1 = densityG1(Ksym,OccNum,Ntot,Kp_x,Kp_y,Gnum,dk,Ct1,Gx,Gy,HXC,out)
                nG = 0.5*(nG0 + nG1)
    #   End of predictor-corrector loop. Ready for the next time step.

        Ct = np.copy(Ct1)
    #------------------------------------------------------------------------------
    #   calculate the excited-state population
    #------------------------------------------------------------------------------
        if Excited_pop == 1:
            N_ex = excited_pop1(Ksym,OccNum,Ntot,Kp_x,Kp_y,Mdim,dk,C,Ct,c)
            if (counter+1)%10==0: print('N_ex =',round(N_ex,12))
            Nex[counter] = N_ex
    #------------------------------------------------------------------------------
    #   calculate the macroscopic current density
    #------------------------------------------------------------------------------
        if Current_mac == 1:
            jx,jy = current1(Ksym,OccNum,Ntot,Kp_x,Kp_y,Mdim,dk,Ct,Gx,Gy,kx,ky)
            jx_mac[counter] = jx - (Ax+Axcx)*Nel/c**2    # add the diamagnetic current
            jy_mac[counter] = jy - (Ay+Axcy)*Nel/c**2    # add the diamagnetic current
            if (counter+1)%10==0: print('jx_mac, jy_mac =',round(jx_mac[counter],12),\
                                                       round(jy_mac[counter],12))
    #------------------------------------------------------------------------------
    #    calculate the time-dependent dielectric constant
    #------------------------------------------------------------------------------
        if Dielectric == 1:
            EPS = epsilon_t(Ksym,OccNum,Ntot,Kp_x,Kp_y,Mdim,dk,kval,Ct,C,ME,E,c,quasi2D)
            if (counter+1)%10==0: print('Eps  =',round(EPS,12))
            Epsilon_t[counter] = EPS
    #------------------------------------------------------------------------------
    #    calculate the time-dependent dipole moment
    #------------------------------------------------------------------------------
        if Dipole == 1:
            DIP = dipole_t(Ksym,OccNum,Ntot,Kp_x,Kp_y,Mdim,dk,Ct,C,ME,E)
            if (counter+1)%10==0: print('Dip  =',np.round(DIP,12))
            Dipole_t[counter] = DIP
    #--------------------------------------------------------------------------
    #             END OF TIME PROPAGATION. OUTPUT OF RESULTS
    #--------------------------------------------------------------------------
    if Excited_pop == 1:
        plt.plot(Time,Nex)
        plt.title('Excited State Population over Time \n E0 = ' + str(E0) + ' alpha_xc = ' + str(alpha_xc))
        plt.xlabel('Time')
        plt.ylabel('Excited State Population')
        plt.xlim([Time[0],Time[-1]])
        plt.show()

        filename = "Nex_E0=" + str(E0) + "alpha_xc=" + str(alpha_xc) + ".txt"
        path = os.path.join(location,filename)

        f = open(path, "w")
        for i in range(Tsteps):
            f.write(str(round(Time[i],5)) + "  " + str(round(Nex[i],12)) + '\n')
        f.close()

    if Current_mac == 1:
        if theta//180 == 0:
            plt.plot(Time,jx_mac,label='Jx')
        elif theta//90 == 0:
            plt.plot(Time,jy_mac,label='Jy')
        else:
            plt.plot(Time,jx_mac,label='Jx')
            plt.plot(Time,jy_mac,label='Jy')
        plt.title('Macroscopic Current Density over Time \n E0 = ' + str(E0) + ' alpha_xc = ' + str(alpha_xc))
        plt.xlabel('Time')
        plt.ylabel('Macroscopic Current Density')
        plt.legend()
        plt.xlim([Time[0],Time[-1]])
        plt.show()

        filename = "j_E0=" + str(E0) + "alpha_xc=" + str(alpha_xc) + ".txt"
        path = os.path.join(location,filename)

        f = open(path, "w")
        for i in range(Tsteps):
            f.write(str(round(Time[i],5)) + "  " + str(round(jx_mac[i].real,12)) + '\n')
        f.close()

    if Dielectric == 1:
        plt.plot(Time,Epsilon_t, label='Epsilon (t)')
        if eps_omega_calc == 1: plt.axhline(epsilon00, label='Epsilon_00', color='black', ls=':')
        plt.title('Dielectric Function over Time \n E0 = ' + str(E0) + ' alpha_xc = ' + str(alpha_xc))
        plt.xlabel('Time')
        plt.ylabel('Dielectric Function')
        plt.xlim([Time[0],Time[-1]])
        plt.legend()
        plt.show()

        filename = "eps_t_E0=" + str(E0) + "alpha_xc=" + str(alpha_xc) + ".txt"
        path = os.path.join(location,filename)

        f = open(path, "w")
        for i in range(Tsteps):
            f.write(str(round(Time[i],6)) + "  " \
                  + str(round(Epsilon_t[i],12)) +  '\n')
        f.close()

    if Dipole == 1:
        plt.plot(Time,Dipole_t)
        plt.title('Dipole Momment over Time \n E0 = ' + str(E0) + ' alpha_xc = ' + str(alpha_xc))
        plt.xlabel('Time')
        plt.ylabel('Dipole Moment')
        plt.xlim([Time[0],Time[-1]])
        plt.show()

        filename = "dip_t_E0=" + str(E0) + "alpha_xc=" + str(alpha_xc) + ".txt"
        path = os.path.join(location,filename)

        f = open(path, "w")
        for i in range(Tsteps):
            f.write(str(round(Time[i],6)) + "  " \
                  + str(round(Dipole_t[i],12)) +  '\n')
        f.close()

    if Occ_final ==1:
        E1 = np.zeros(Kp_x*Kp_y*Ntot)
        F1 = np.zeros(Kp_x*Kp_y*Ntot)
        F_occ = Fermi_dist(Ksym,OccNum,Ntot,Kp_x,Kp_y,Mdim,dk,C,Ct)
        for i in range(Kp_x*Kp_y*Ntot):
            E1[i] = Eaux[sorted[i]]
            F1[i] = F_occ[sorted[i]]

        plt.plot(E1,F1,'.')
        plt.title('Final Occupation Numbers of the Energy States')
        plt.xlabel('Energy')
        plt.ylabel('Final Occupation Number')
    #   plt.xlim([0,Ntot])
        plt.show()
