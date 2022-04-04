# -*- coding: utf-8 -*-
"""
Created on Fri October 29 2021

@author: Elliot Skey
"""



import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import HMCCSFUC as UC
import unewfuncs as unew
from scipy import stats

plt.rcParams["mathtext.fontset"] = 'cm' #plot font aesthetics
plt.rcParams.update({'font.size': 12})

def main():
    #input and inititalise values
    discards = int(input("How many sweeps at beginning discarded: "))
    sites = int(input("how many lattice sites: "))
    mass = input("input mass parameter: ")
    stepsize = float(input("input stepsize: "))
    steps = int(input("number of steps in update: "))
    maxdt = int(sites/2+1)
    field = np.array([[complex(np.random.rand(),np.random.rand()) for i in range(sites)] for j in range(sites)])
    sweeps = int(input("how many sweeps?: "))
    binwidth = int(input("input bin width for jackknife analysis, (must divide number of sweeps-discards): "))
    mom = [0]
    system = UC.configuration(field,mom,mass,stepsize,steps) #initialise system with update class
    sqaveragepaths = []
    meanpaths = []
    fourthaverage = []
    actions = []
    twopointarray = [[0 for i in range((int(sweeps-discards)))] for j in range(maxdt)]
    
    #twopointfile = UC.createfile(f'm{mass}sites{sites}sweeps{sweeps}twopoint.dat')
    twopointtimeslicefile  = UC.createfile('m{mass}sites{sites}sweeps{sweeps}twopointtimeslice.dat')
    datafile = UC.createfile(f'm{mass}sites{sites}sweeps{sweeps}datafile.dat')
    for i in tqdm.trange(sweeps,mininterval = 1, desc = "progress through hmc sweeps"): #loop through sweeps of hmc algorithm
        #twopointarray = []
        twopointtimeslicearray = []
        system.HMCUp() #update path using hmc alg
        sqav = np.real(system.meansquare())
        sqaveragepaths.append(sqav)
        meanav = system.mean()
        meanpaths.append(meanav)
        fourav = np.real(system.meanfour())
        fourthaverage.append(fourav)
        if i >= discards:                           #record data, write to file etc
            datafile = open(f'm{mass}sites{sites}sweeps{sweeps}datafile.dat','a')
            #twopointfile = open(f'm{mass}sites{sites}sweeps{sweeps}twopoint.dat','a')    
            twopointtimeslicefile = open(f'm{mass}sites{sites}sweeps{sweeps}twopointtimeslice.dat','a')
            action = system.totalaction()
            for j in range(maxdt):
               #twopointarray.append(np.real(UC.twopoint(system,j)))
               twopts = np.real(UC.twopointtimeslice(system.field,j))
               twopointtimeslicearray.append(twopts)
            #np.savetxt(twopointfile,[twopointarray])
            np.savetxt(twopointtimeslicefile,[twopointtimeslicearray])
            #twopointfile.close()
            twopointtimeslicefile.close()
            UC.writedata([sqav,np.real(meanav),np.imag(meanav),fourav,np.real(action),np.real(system.data[-1])], datafile)
            datafile.close()
            actions.append(action)
    
    twopoint = UC.datafromfile(f'm{mass}sites{sites}sweeps{sweeps}twopoint.dat',binwidth,discards)
    twopointtotal = twopoint['func']
    twopointerror = twopoint['funcerror']
    twopointtimeslice = UC.datafromfile('m{mass}sites{sites}sweeps{sweeps}twopointtimeslice.dat',binwidth,discards)
    twopointtimeslicetotal = twopointtimeslice['func']
    twopointtimesliceerror = twopointtimeslice['funcerror']
    logtpts = np.log(twopointtimeslicetotal[:4])
    corlendata = stats.linregress(range(4),logtpts)
    print()
    #print(twopointtimeslicetotal)
    print()
    
    print()
    print('acceptance rate = ', system.accepted/sweeps)
    print('discard rate = ', system.discarded/sweeps)
    print()
    
    print()
    alldata = unew.datas(f"{Path.cwd()}\\m{mass}sites{sites}sweeps{sweeps}datafile.dat")
    print()

    #load data into dicts for organising\printing purposes

    unbiasphi = [alldata.value[1],alldata.dvalue[1],alldata.tau_int[1],alldata.dtau_int[1]]
    phidata = UC.dictify(np.real(np.array(meanpaths)),unbiasphi,discards,binwidth,'Re(phi)')
    unbiasphiim = [alldata.value[2],alldata.dvalue[2],alldata.tau_int[2],alldata.dtau_int[2]]
    phiimdata = UC.dictify(np.real(np.array(meanpaths)),unbiasphiim,discards,binwidth,'Im(phi)')

    unbiasphi2 = [alldata.value[0],alldata.dvalue[0],alldata.tau_int[0]]
    phi2data = UC.dictify(sqaveragepaths,unbiasphi2,discards,binwidth,'phi^2')
    unbiasphi4 = [alldata.value[3],alldata.dvalue[3],alldata.tau_int[3]]
    phi4data = UC.dictify(fourthaverage,unbiasphi4,discards,binwidth,'phi^4')
    unbiasaction = [alldata.value[4],alldata.dvalue[4],alldata.tau_int[4]]
    actiondata = UC.dictify(actions,unbiasaction,0,binwidth,'S/N**2')
    unbiasdham = [alldata.value[5],alldata.dvalue[5],alldata.dvalue[5],alldata.tau_int[5]]
    dhamdata = UC.dictify(system.data,unbiasdham,discards,binwidth,'e^(-dham)')
    
    
    #unbiascorlen = [alldata.value[6],alldata.dvalue[6],alldata.dvalue[6],alldata.tau_int[6]]
    #corlendata = UC.dictify(corlen,unbiascorlen,discards,binwidth,'corlen')
    
    
    
    print()
    print('acceptance rate = ', system.accepted/sweeps)
    print('discard rate = ', system.discarded/sweeps)
    print()
    
    UC.printdata(phidata)
    UC.printdata(phiimdata)
    UC.printdata(phi2data)
    print(f'error in <phi^2> autocorrelation time = {alldata.dtau_int[0]}')
    UC.printdata(phi4data)
    UC.printdata(actiondata)
    UC.printdata(dhamdata)
    #UC.printdata(corlendata)
    print()
    print(f'correlation length = {-corlendata[0]}')
    print(f'error in correlation length= {corlendata[4]}')
    print()
    
    """
    f = plt.figure(0)
    f, plots = plt.subplots(2,2) #plot various data
    f.suptitle(r'Plot of $\langle\varphi^2\rangle, \langle\varphi\rangle, \langle\varphi^4\rangle$ against Monte Carlo Time')
    plots[0].plot(sqaveragepaths,'.',markevery=20)
    plots[0].set_title(r'Plot of $\langle\varphi^2\rangle$')
    plots[1].plot(meanpaths,'.', markevery=20)
    plots[1].set_title(r'Plot of $\langle\varphi\rangle$')
    plots[2].set_title(r'Plot of $\langle\varphi^4\rangle$')
    plots[2].plot(fourthaverage,'.',markevery=20)
    plt.savefig(f'sweeps{sweeps}sites{str(sites)}m{mass}plot1.pdf',format='pdf')
    f.show()
    """
    g = plt.figure(1)
    g, plots1 = plt.subplots(1)
    g.suptitle(r'Time Sliced Correlator $G(t)$')
    """
    plots1[0].set_title(r'$G(d\tau)$')
    plots1[0].errorbar([i for i in range(len(twopointtotal))],twopointtotal, fmt = '.', yerr = list(twopointerror),capsize = 3)
    plots1[0].set_yscale('log')
    
    plots1[1].plot(np.real(system.data),'.',markevery = 20)
    plots1[1].set_title(r'$e^{-\Delta\mathcal{H}}$')
    """
    plots1.errorbar([i for i in range(len(twopointtimeslicetotal))],twopointtimeslicetotal, fmt = '.', yerr = list(twopointtimesliceerror),capsize = 3)
    plots1.set_yscale('log')
    plt.savefig(f'sweeps{sweeps}sites{str(sites)}m{mass}correlatorplot.pdf',format='pdf')
    g.show()
    
    input()
    
if __name__ == "__main__":
    main()
        
