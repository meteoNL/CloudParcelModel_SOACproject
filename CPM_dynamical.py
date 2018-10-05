# -*- coding: utf-8 -*-
"""
Created on Fri Oct 05 08:12:20 2018

@author: Edward
"""

import numpy as np
import matplotlib.pyplot as pl
g=9.81
cp=1005
tend=7200.
dt=0.1
t1=np.linspace(1,tend,(tend/dt))

Tp=305 #temp air parcel, K
zp=5
w=0

data_env=np.array([])
data_p=np.array([])

f=open('dataset.txt','r')
i=0
for line in f:
    line=line.split(';')
    print(line[2])
    z=float(line[2])
    T=float(line[3])+273.15
    data_env=np.append(data_env,np.array([z,T]))
    i+=1
data_env=np.reshape(data_env,(int(len(data_env)/2.0),2))

def dwdt(Tp,Tenv):
    return g*(Tp-Tenv)/Tenv

def dTpdt(w):
    return -g*w/cp

def Tenv(z):
    j=0
    while data_env[j,0] < z:
        Tenv_plus1=data_env[(j+1),1]
        Tenv_0=data_env[j,1]
        zenv_plus1=data_env[(j+1),0]
        zenv_0=data_env[j,0]
        dTdz=(Tenv_plus1-Tenv_0)/(zenv_plus1-zenv_0)
        T=Tenv_0+(z-zenv_0)*dTdz
        j+=1
    return T

for t in t1:
    Tenv_new=Tenv(zp)
    w=w+dwdt(Tp,Tenv_new)
    zp=zp+w*dt
    Tp=Tp+dTpdt(w)
    data_p=np.append(data_p,np.array([zp,Tp,w]))

data_p=np.reshape(data_p,(int(len(data_p)/3.0),3))
pl.plot(data_env[:,1],data_env[:,0])
pl.plot(data_p[:,1],data_p[:,0])
pl.show()

data=np.zeros((3,len(t1)+1))


for t in t1:
#    w=wnew(thetap,z,w)
 #   z=znew(thetap,z,w)
 #   print(w,z,t)   
    data[0,int(t/dt)]=z
    data[1,int(t/dt)]=w
    data[2,int(t/dt)]=t
    



