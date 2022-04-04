# -*- coding: utf-8 -*-
"""
Created on Fri October 29 2021

@author: Elliot Skey
"""

import tqdm
import numpy as np
import matplotlib.pyplot as plt
import HMCFAPhi4UC as UC
from pathlib import Path
import unewfuncs as unew
from scipy import stats
from numpy import fft

plt.rcParams["mathtext.fontset"] = 'cm' #plot font aesthetics
plt.rcParams.update({'font.size': 12})

def realpropcalc(N,mass):
    sinp = [np.sin(np.pi*i/N)**2 for i in range(N)]
    Gfourier = np.array([[1/(4*(sinp[i]+sinp[j])+mass**2) for i in range(N)] for j in range(N)])
    stdev = [[np.sqrt(N**2/(2*Gfourier[i][j])) for i in range(N)]for j in range(N)]
    G = fft.ifft2(Gfourier)
    Gtimeslice = np.sum(G,axis=0)
    return Gfourier, stdev, Gtimeslice[:int(N/2)]


def main():
    #input and inititalise values
    discards = int(input("how many initial discarded: "))
    sites = int(input("how many lattice sites: "))
    mass = float(input("input mass parameter: "))
    kappa = float(input("input fourier kappa parameter: "))
    lam = float(input("input interaction parameter: "))
    stepsize = float(input("input stepsize: "))
    steps = int(input("number of steps in update: "))
    maxdt = int(sites/2+1)
    field = np.array([[complex(np.random.rand(),np.random.rand()) for j in range(sites)] for i in range(sites)])
    sweeps = int(input("how many sweeps?: "))
    binwidth = int(input("input bin width for jackknife analysis: "))
    mom = [0]
    system = UC.configuration(field,mom,mass,kappa,stepsize,steps,lam) #initialise system with update class
    sqaveragepaths = []
    meanpaths = []
    fourthaverage = []
    actions = []
    #corlen = []
    
    twopointfile = UC.createfile(f'sweeps{sweeps}m{mass}kappa{kappa}{lam}twopoint.dat')
    twopointtimeslicefile  = UC.createfile(f'sweeps{sweeps}m{mass}kappa{kappa}{lam}twopointtimeslice.dat')
    datafile = UC.createfile(f'm{mass}kap{kappa}lam{lam}sites{sites}sweeps{sweeps}datafile.dat')
    for i in tqdm.trange(sweeps,mininterval = 1, desc = "progress through hmc sweeps"): #loop through sweeps of hmc algorithm
        twopointarray = []
        twopointtimeslicearray = []
        system.HMCUp() #update path using hmc alg
        sqav = np.real(system.meansquare())
        sqaveragepaths.append(sqav)
        meanav = system.mean()
        meanpaths.append(meanav)
        fourav = np.real(system.meanfour())
        fourthaverage.append(fourav)
        if i >= discards:                           #record data, write to file etc
            datafile = open(f'm{mass}kap{kappa}lam{lam}sites{sites}sweeps{sweeps}datafile.dat','a')
            twopointfile = open(f'sweeps{sweeps}m{mass}kappa{kappa}{lam}twopoint.dat','a')    
            twopointtimeslicefile = open(f'sweeps{sweeps}m{mass}kappa{kappa}{lam}twopointtimeslice.dat','a')
            action = system.totalaction()
            for j in range(maxdt):
               twopointarray.append(np.real(UC.twopoint(system,j)))
               twopts = np.real(UC.twopointtimeslice(system.field,j))
               twopointtimeslicearray.append(twopts)
            np.savetxt(twopointfile,[twopointarray])
            np.savetxt(twopointtimeslicefile,[twopointtimeslicearray])
            twopointfile.close()
            twopointtimeslicefile.close()
            UC.writedata([sqav,np.real(meanav),np.imag(meanav),fourav,np.real(action),np.real(system.data[-1])], datafile)
            datafile.close()
            actions.append(action)
    
    twopoint = UC.datafromfile(f'sweeps{sweeps}m{mass}kappa{kappa}{lam}twopoint.dat',binwidth,discards)
    twopointtotal = twopoint['func']
    twopointerror = twopoint['funcerror']
    twopointtimeslice = UC.datafromfile(f'sweeps{sweeps}m{mass}kappa{kappa}{lam}twopointtimeslice.dat',binwidth,discards)
    twopointtimeslicetotal = twopointtimeslice['func']
    twopointtimesliceerror = twopointtimeslice['funcerror']
    logtpts = np.log(twopointtimeslicetotal[:4])
    corlendata = stats.linregress(range(4),logtpts)
    print()
    print(twopointtimeslicetotal)
    print()
    
    print()
    print('acceptance rate = ', system.accepted/sweeps)
    print('discard rate = ', system.discarded/sweeps)
    print()
    
    print()
    alldata = unew.datas(f"{Path.cwd()}\\m{mass}kap{kappa}lam{lam}sites{sites}sweeps{sweeps}datafile.dat")
    print()

    #load data into dicts for organising\printing purposes

    unbiasphi = [alldata.value[1],alldata.dvalue[1],alldata.tau_int[1],alldata.dtau_int[1]]
    phidata = UC.dictify(np.real(np.array(meanpaths)),unbiasphi,discards,binwidth,'Re(phi)')
    unbiasphiim = [alldata.value[2],alldata.dvalue[2],alldata.tau_int[2],alldata.dtau_int[2]]
    phiimdata = UC.dictify(np.imag(np.array(meanpaths)),unbiasphiim,discards,binwidth,'Im(phi)')
    unbiasphi2 = [alldata.value[0],alldata.dvalue[0],alldata.tau_int[0],alldata.dtau_int[0]]
    phi2data = UC.dictify(sqaveragepaths,unbiasphi2,discards,binwidth,'phi^2')
    unbiasphi4 = [alldata.value[3],alldata.dvalue[3],alldata.tau_int[3],alldata.dtau_int[3]]
    phi4data = UC.dictify(fourthaverage,unbiasphi4,discards,binwidth,'phi^4')
    unbiasaction = [alldata.value[4],alldata.dvalue[4],alldata.tau_int[4],alldata.dtau_int[4]]
    actiondata = UC.dictify(actions,unbiasaction,0,binwidth,'S/N**2')
    unbiasdham = [alldata.value[5],alldata.dvalue[5],alldata.tau_int[5],alldata.dtau_int[5]]
    dhamdata = UC.dictify(system.data,unbiasdham,discards,binwidth,'e^(-dham)')
    #unbiascorlen = [alldata.value[6],alldata.dvalue[6],alldata.dvalue[6],alldata.tau_int[6]]
    #corlendata = UC.dictify(corlen,unbiascorlen,discards,binwidth,'corlen')
    print()
    print('acceptance rate = ', system.accepted/sweeps)
    print('discard rate = ', system.discarded/sweeps)
    print()
    
    UC.printdata(phidata)
    UC.printdata(phiimdata)
    #print(f'naive error in Im<phi> = {UC.mean(np.imag(np.array(meanpaths)),discards)}')
    #print(f'jackknife error in Im<phi> = {UC.jackknifevar(np.imag(np.array(meanpaths)),binwidth,discards)}')
    
    
    UC.printdata(phi2data)
    #print(f'error in <phi^2> autocorrelation time = {alldata.dtau_int[0]}')
    UC.printdata(phi4data)
    UC.printdata(actiondata)
    UC.printdata(dhamdata)
    
    #UC.printdata(corlendata)
    
    print()
    print(f'correlation length = {-corlendata[0]}')
    print(f'error in correlation length= {corlendata[4]}')
    print()
    
    f = plt.figure(0)
    f, plots = plt.subplots(1) #plot various data
    plots.plot(sqaveragepaths,'.',markevery=20)
    #plots.set_title(r'$\langle\varphi^2\rangle$')
    f.supxlabel('Metropolis Sweeps')
    plt.savefig(f'sweeps{sweeps}{str(sites)}m{mass}kap{kappa}lam{lam}plotphi2.pdf',format='pdf')
    plt.clf()
    
    f = plt.figure(0)
    f, plots = plt.subplots(1) #plot various data
    plots.plot(fourthaverage,'.',markevery=20)
    #plots.set_title(r'$\langle\varphi^4\rangle$')
    f.supxlabel('Metropolis Sweeps')
    plt.savefig(f'sweeps{sweeps}{str(sites)}m{mass}kap{kappa}lam{lam}plotphi4.pdf',format='pdf')
    plt.clf()
    
    f = plt.figure(0)
    f, plots = plt.subplots(1) #plot various data
    plots.plot(meanpaths,'.',markevery=20)
    #plots.set_title(r'$Re\langle\varphi\rangle$')
    f.supxlabel('Metropolis Sweeps')
    plt.savefig(f'sweeps{sweeps}{sites}m{mass}kap{kappa}lam{lam}plotphi.pdf',format='pdf')
    plt.clf()

    
    
    """
    g = plt.figure(1)
    g, plots1 = plt.subplots(3)
    g.suptitle(r'Correlator as function of $d\tau$ & $e^{-\Delta\mathcal{H}}$ against Monte Carlo Time')
    plots1[0].set_title(r'$G(d\tau)$')
    plots1[0].errorbar([i for i in range(len(twopointtotal))],twopointtotal, fmt = '.', yerr = list(twopointerror),capsize = 3)
    plots1[0].set_yscale('log')
    plots1[1].plot(np.real(system.data),'.',markevery = 20)
    plots1[1].set_title(r'$e^{-\Delta\mathcal{H}}$')
    plots1[2].errorbar([i for i in range(len(twopointtimeslicetotal))],twopointtimeslicetotal, fmt = '.', yerr = list(twopointtimesliceerror),capsize = 3)
    plots1[2].set_yscale('log')
    plt.savefig(f'{str(sites)}m{mass}kap{kappa}plot2.pdf',format='pdf')
    g.show()
    """
    
    prop = realpropcalc(sites,mass)[2]
    print(prop)
    fig = plt.figure()
    fig, plots = plt.subplots(1) #plot various data
    plots.plot(prop)
    plots.errorbar(range(len(twopointtimeslicetotal)),twopointtimeslicetotal,yerr = twopointtimesliceerror,fmt  = '.',capsize = 4)
    plots.set_yscale('log') 
    plots.set_title(r'G(t)')
    fig.supxlabel('t')
    plt.savefig(f'mass{mass}lam{lam}sweeps{sweeps }hmcfreetimeslliceFreeCorrelator.pdf',format = 'pdf')
    plt.clf()
    input()
    
if __name__ == "__main__":
    main()