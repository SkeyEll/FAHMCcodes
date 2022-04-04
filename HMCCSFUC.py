# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 14:41:13 2021

@author: Elliot
"""

#Notes P Pdag here since it doesn't matter

#doesn't want to work for lower acceptance rates

import numpy as np
import random as ran
import astropy.stats.jackknife as jk

plusmin = u"\u00B1"

##p = pstar everywhere


#data organising functions
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

def printdata(obs):
    print()
    print('naive mean <' + obs['label']+ '> = ' + str(obs['mean']))
    print('naive error in <' + obs['label'] + '> = ' + str(obs['naiveerror']))
    print('jackknife error in <' +obs['label']+ '> = ' + str(obs['jackerror']))
    print('unbiased ' +obs['label'] + ' = ' + str(obs['unbiasmean']) + plusmin + str(obs['unbiaserror']))
    print('int autocorrelation time of <' +obs['label']+ '> = ' + str(obs['intcor']))
    print()
    return

def writedata(obs,file):
    np.savetxt(file, [obs], delimiter = '\t')
    return

def errorstring(obs,error):
    return str(obs) + u"\u00B1" + str(error)






#physics type functions
def NNsum(field): #sum of nearest neighbour for a field for each site
    NN = 0
    for k in [1,-1]:
        NN += np.roll(field,k,axis=0)+np.roll(field,k,axis=1)
    return NN

def leapfrog(N,e,l,phi,p,m):#leapfrog method with half step on phi
    phi = phi + 0.5*e*p    
    for k in range(l-1):
        p = p - e*((m**2+4)*phi-NNsum(phi))
        phi = phi + e*p    
    p = p - e*((m**2+4)*phi-NNsum(phi))
    phi = phi + 0.5*e*p   
    return phi,p
    
def totalHamiltonian(m,phi,p):#hamiltonian for system
    return np.sum(p*np.conj(p)+np.conj(phi)*((m**2+4)*phi-NNsum(phi)))

def twopoint(system,distance):#two point function
    phi = system.field
    phidt = np.roll(phi,distance,axis = 0) + np.roll(phi,distance,axis = 1)
    G = np.sum(np.multiply(np.conj(phi),phidt))
    return G/(2*(len(phi))**2)


def twopointtimeslice(phi,dt):
    phidt = np.roll(sum(np.roll(phi,i,axis = 0) for i in range(len(phi))),dt,axis = 1)
    return np.sum(np.conj(phi)*phidt)/len(phi)**2

def corlen(twopointfunc):
    cor = -np.log(np.abs(twopointfunc[2]/twopointfunc[0]))
    return cor/2


#statistical functions
def mean(observable,discards):#mean without discards
    obs = observable[discards:]
    return np.average(obs)

def stdev(observable,discards): #returns naive error of a set of points(mean values after a sweep) starting at the discardsth sweep
    obs = observable[discards:]
    return np.std(obs)/np.sqrt(len(obs))


def jackknifevar(obs,B,discards): #standard jackknife error for binssize B
    bins = obs[discards:]
    bins = np.array_split(bins,np.floor(len(bins)/B))
    jackstats = np.array([np.sum(group)/len(group) for group in bins])
    datas = jk.jackknife_stats(jackstats,np.mean)
    return datas[2]


def jackknifevarother(obs,B,discards): #jackknife method for error has a 
    bins = obs[discards:]                                   #new list of obs minus discards
    bins = np.array_split(bins,np.floor(len(bins)/B))       #split into equal size bins
    binav = []
    for group in bins:
        binav.append(np.average(group))                     #average bins
    return np.std(np.array(binav))/np.sqrt(len(bins))       #error in bin averages


#Update class - contains system and related functions 
class configuration(object): #add kinetic energy method

    def __init__(self,conf,mom,mass,stepsize,steps):
        self.field = np.array(conf)
        self.mass = float(mass)
        self.stepsize = float(stepsize)
        self.steps = int(steps)
        self.accepted = 0
        self.discarded = 0
        self.data = []
        
    def __str__(self):
        return str(self.mass)
            
    def totalaction(self):#return action per site
        m = self.mass
        phi = self.field
        N = self.numsites()
        S =  np.sum((m**2+4)*np.multiply(np.conj(phi),phi) - np.multiply(np.conj(phi),NNsum(phi)))
        return S/N**2
    """
    def kineticenergy(self): #kinetic energy per site
        p = self.mom
        return np.average(p*np.conj(p))
    """
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
        p = np.array([[complex(np.random.normal(0,np.sqrt(1/2)),np.random.normal(0,np.sqrt(1/2))) for j in range(self.numsites())] for i in range(self.numsites())])
        phi, m, N, L, e = self.field, self.mass, self.numsites(), self.steps, self.stepsize
        initialH = totalHamiltonian(m,phi,p) #calculate hamiltonian of new initial config
        updatedsystem = leapfrog(N,e,L,phi,p,m) #update field and momentum
        updatedfield = updatedsystem[0]
        updatedmom = updatedsystem[1]
        finalH = totalHamiltonian(m,updatedfield,updatedmom)
        dHam = finalH-initialH
        exp = np.exp(-dHam)
        self.data.append(exp)
        r = np.random.rand()
        if r < exp:
            self.accepted = self.accepted+1
            self.field = updatedfield
        else:
            self.discarded += 1