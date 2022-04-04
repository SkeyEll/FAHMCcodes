# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 14:41:13 2021

@author: Elliot
"""

import numpy as np
import random as ran
from numpy import fft
from numba import njit
import astropy.stats.jackknife as jk

plusmin = u"\u00B1"

def createfile(filename):
    file = open(filename,'w')
    file.close()

def datafromfile(filename,binwidth,discards):
    file = open(filename,'r')
    data = np.transpose(np.loadtxt(file))
    datadict = {}
    datadict['func'] = [mean(dt,0) for dt in data]
    print(datadict['func'])
    datadict['funcerror'] = [jackknifevar(dt,binwidth,discards) for dt in data]
    return datadict


def dictify(obs,unbias,discards,binwidth,label):
    dickt = {}
    dickt['label'] = label
    dickt['mean'] = mean(obs,discards)
    dickt['naiveerror'] = stdev(obs,discards)
    dickt['jackerror'] = jackknifevar(obs, binwidth, discards)
    
    """
    if len(unbias) == 6: #if observable is complex valued
        dickt['unbiasmean'] = complex(unbias[0],unbias[1])
        dickt['unbiaserror'] = complex(unbias[2],unbias[3])
        dickt['intcor'] = complex(unbias[4],unbias[5])
    """
    dickt['unbiasmean'] = unbias[0]
    dickt['unbiaserror'] = unbias[1]
    dickt['intcor'] = unbias[2]
    dickt['intcorerror'] = unbias[3]
    return dickt

def printdata(obs):
    print()
    print(f"naive mean <{obs['label']}> = {obs['mean']}")
    print(f"naive error in <{obs['label']}> = {obs['naiveerror']}")
    print(f"jackknife error in <{obs['label']}> = {obs['jackerror']}")
    print(f"unbiased {obs['label']} = {obs['unbiasmean']} {plusmin} {obs['unbiaserror']}")
    print(f"int autocorrelation time of <{obs['label']}> = {obs['intcor']} {plusmin} {obs['intcorerror']}")
    print()
    return

def writedata(obs,file):
    np.savetxt(file, [obs], delimiter = '\t')

def errorstring(obs,error):
    return f"{obs} {plusmin} {error}"

def NNsum(field):
    return sum(np.roll(field,k,axis=j)for k in [1,-1] for j in range(2))

@njit
def NNsumNjit(field): #gets nearest neighbour sites for all sites on lattice for a given field~~not exactly, its sums the ones in positive direcs and then negative dirctions
    NN = np.full((len(field),len(field)),complex(0,0))
    for k in [1,-1]:
        NN += np.roll(field,k)+np.transpose(np.roll(np.transpose(field),k))
    return NN

def propcalc(N,kappa):
    sinp = [np.sin(np.pi*i/N)**2 for i in range(N)]
    Gfourier = np.array([[1/(4*kappa*(sinp[i]+sinp[j])+1-kappa) for i in range(N)] for j in range(N)])
    stdev = [[np.sqrt(N**2/(2*Gfourier[i][j])) for i in range(N)]for j in range(N)]
    G = fft.ifft2(Gfourier)
    Gtimeslice = np.sum(G,axis=0)
    return Gfourier, stdev, Gtimeslice

def genfourmom(N,stdev):
    p = np.array([[complex(np.random.normal(0,stdev[i][j]),np.random.normal(0,stdev[i][j])) for i in range(N)] for j in range(N)])  #generate fourier ps
    return p

def upphi(e,pi,phi,propcalc):
    field = fft.fft2(phi)
    pi = fft.fft2(np.conj(pi))
    field = field + e*pi*propcalc
    return fft.ifft2(field)

def leapfrogint(e,L,phi,p,m,prop,lam): #interacting case leapfrog
    p = fft.ifft2(p)
    p = p - 0.5*e*((m**2+4)*np.conj(phi) - np.conj(NNsum(phi))+lam/3*np.conj(phi)**2*phi)
    for k in range(L-1):
        phi = upphi(e,p,phi,prop)
        p = p - e*((m**2+4)*np.conj(phi) - np.conj(NNsum(phi))+lam/3*np.conj(phi)**2*phi)
    phi = upphi(e,p,phi,prop)
    p = p - 0.5*e*((m**2+4)*np.conj(phi) - np.conj(NNsum(phi))+lam/3*np.conj(phi)**2*phi)
    return phi,p

def leapfrogfree(e,L,phi,p,m,prop): #free case leapfrog
    p = fft.ifft2(p)
    p = p - 0.5*e*((m**2+4)*np.conj(phi) - np.conj(NNsum(phi)))
    for k in range(L-1):
        phi = upphi(e,p,phi,prop)
        p = p - e*((m**2+4)*np.conj(phi) - np.conj(NNsum(phi)))
    phi = upphi(e,p,phi,prop)
    p = p - 0.5*e*((m**2+4)*np.conj(phi) - np.conj(NNsum(phi)))
    return phi,p



def totalHamiltonian(m,phi,p,n,prop,lam):
    phi2 = np.conj(phi)*phi
    if lam == 0: #calculate less if free case, used for efficiency
        phi4 = 0
    else:
        phi4 = phi2**2
    return np.sum(1/(n**2)*np.conj(p)*prop*p + (m**2+4)*phi2 + lam/6*phi4  - np.conj(phi)*NNsum(phi))

@njit
def totalHamiltonianNjit(m,phi,p,n,prop,lam): 
    N = len(phi)
    phi2 = np.multiply(np.conj(phi),phi)
    if lam == 0: #calculate less if free case, used for efficiency
        phi4 = np.full((N,N),complex(0,0))
    else:
        phi4 = np.multiply(phi2,phi2)
    H = np.sum(np.full((N,N),complex(1/(n**2),0))*np.multiply(np.multiply(np.conj(p),prop),p) + np.full((N,N),complex((m**2+4),0))*phi2 + np.full((N,N),complex(lam/6,0))*phi4  - np.multiply(np.conj(phi),NNsum(phi)))
    return H

def twopoint(system,distance): #perpendicular correlation function
    phi = system.field
    phidt = np.roll(phi,distance,axis = 0) + np.roll(phi,distance,axis = 1) #sums over both axes
    G = np.sum(np.multiply(np.conj(phi),phidt)) 
    return G/(2*(len(phi))**2)

@njit
def twopointtimeslicenjit(phi,dt):#time sliced correlation function
    N = len(phi)
    phidt = np.full((N,N),complex(0,0))
    for i in range(N):
        phidt += np.roll(phi,i)
    phidt = np.transpose(np.roll(np.transpose(phidt),dt))
    Gtot = np.multiply(np.conj(phi),phidt)
    return np.sum(Gtot)/N**2

def twopointtimeslice(phi,dt):
    phidt = np.roll(sum(np.roll(phi,i,axis = 0) for i in range(len(phi))),dt,axis = 1)
    return np.sum(np.conj(phi)*phidt)/len(phi)**2
    
def corlen(twopointfunc):
    """maxdt = int(np.floor(2*len(twopointfunc)/10)) #simple choice so that we can solve for inital gradient of the propagator automatically
    effmass = 0
    for i in range(maxdt-1):
        effmass += np.log(twopointfunc[i+1]/twopointfunc[i]) #gradient on log graph"""
    cor = -np.log(np.abs(twopointfunc[2]/twopointfunc[0]))
    return cor/2
def mean(observable,discards):
    obs = observable[discards:]
    return np.sum(obs)/len(obs)

def stdev(observable,discards): #returns standard deviation of a set of points(mean values after a sweep) starting at the discardsth sweep
    obs = observable[discards:]
    return np.std(obs)/np.sqrt(len(obs))


def jackknifevar(obs,B,discards): #standard jackknife error for binssize B
    bins = obs[discards:]
    bins = np.array_split(bins,np.floor(len(bins)/B))
    jackstats = np.array([np.sum(group)/len(group) for group in bins])
    datas = jk.jackknife_stats(jackstats,np.mean)
    return datas[2]

def jackknifevarcorrectbutbinsizenminus1(obs,B,discards):
    bins = np.array(obs[discards:])
    N = len(bins)
    meanobs = np.mean(bins)
    jkvar = meanobs-bins
    return np.sqrt(np.sum(jkvar**2)/(N*(N-1)))

def jackknifevarjustbins(obs,B,discards): #not really correct but close enough less expensive than the other one
    bins = obs[discards:]                             #new list of obs minus discards
    bins = np.array_split(bins,np.floor(len(bins)/B)) #split into equal size bins
    binav = [np.average(group) for group in bins]
    return np.std(np.array(binav))/np.sqrt(len(bins))          #error in bin averages

class configuration(object): #add kinetic energy method

    def __init__(self,conf,mom,mass,kappa,stepsize,steps,lam):
        self.field = np.array(conf)
        self.mass = float(mass)
        self.kappa = float(kappa)
        self.stepsize = float(stepsize)
        self.steps = int(steps)
        self.accepted = 0
        self.discarded = 0
        self.data = []
        self.fourprop = propcalc(len(self.field), kappa)
        self.lam = lam
        
    def __str__(self):
        return self.field

    def totalaction(self):
        m = self.mass
        phi = self.field
        N = self.numsites()
        lam = self.lam
        phi2 = np.multiply(np.conj(phi),phi)
        if lam == 0: #calculate less if free case, used for efficiency
            phi4 = 0
        else:
            phi4 = phi2**2
        S =  np.sum((m**2+4)*phi2 + lam/6*phi4  - np.conj(phi)*NNsum(phi))
        return S/N**2
    
    def numsites(self):
        return int(len(self.field)) #returns number of lattice sites

    def meansquare(self):
        return np.sum(np.multiply(self.field,np.conj(self.field)))/self.numsites()**2

    def mean(self):
        return np.sum(self.field)/self.numsites()**2
    
    def meanfour(self):
        phi2 = np.multiply(self.field,np.conj(self.field))
        return np.sum(np.multiply(phi2,phi2))/(self.numsites()**2)
       
    def HMCUp(self): #integrate ham eqns then perform metrpolis update on config
        phi, m, N, L, e, fourprop, lam = self.field, self.mass, self.numsites(), self.steps, self.stepsize, self.fourprop, self.lam
        p =  genfourmom(N,fourprop[1])
        initialH = totalHamiltonian(m,phi,p,N,fourprop[0],lam) #calculate hamiltonian of new initial config
        if lam == 0:
            updatedsystem = leapfrogfree(e,L,phi,p,m,fourprop[0]) #update field and momentum
        else:   
            updatedsystem = leapfrogint(e,L,phi,p,m,fourprop[0],lam)
        finalH = totalHamiltonian(m,updatedsystem[0],fft.fft2(updatedsystem[1]),N,fourprop[0],lam)
        dHam = finalH-initialH
        exp = np.exp(-dHam)
        self.data.append(exp)
        r = ran.random()
        if r < exp:
            self.accepted = self.accepted+1
            self.field = updatedsystem[0]
        else:
            self.discarded += 1