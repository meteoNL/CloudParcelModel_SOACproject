# -*- coding: utf-8 -*-
"""
Created on Fri Oct 05 08:12:20 2018

@author: Edward chenxi
"""

import numpy as np
import matplotlib.pyplot as pl
g=9.81
cp=1005
tend=3600.
dt=0.5
gamma=0.5
mu=5e-4

t1=np.linspace(0.0,tend,int(tend/dt)) 

Tp=308 #temp air parcel, K
zp=170
w=0
wvp=15.0

data_env=np.array([])
data_p=np.array([])

f=open('dataset.txt','r')
i=0
for line in f:
    line=line.split(';')
    z=float(line[2])
    T=float(line[3])+273.15
    wv=float(line[6])
    data_env=np.append(data_env,np.array([z,T,wv]))
    i+=1
data_env=np.reshape(data_env,(int(len(data_env)/3.0),3))
f.close()
def dwdt(Tp,Tenv):
    return 1/(1+gamma)*(g*(Tp-Tenv)/Tenv-mu*abs(w)*w)

def dTpdt(w,Tp,zp):
    return -(g*w/cp+mu*abs(w)*(Tp-Tenv(zp)))

def dwvdt(w,wvp,wvenv):
    return -mu*(wvp-wvenv)*abs(w)
    

def Tenv(z):
    j=0
    if data_env[j,0] == z or data_env[j,0] > z: #to prevent it from leaving domain
        T=data_env[0,1]
    elif data_env[-1,0] < z: #to prevent it from leaving the domain
        T=300
    while data_env[j,0] < z:
        Tenv_plus1=data_env[(j+1),1]
        Tenv_0=data_env[j,1]
        zenv_plus1=data_env[(j+1),0]
        zenv_0=data_env[j,0]
        dTdz=(Tenv_plus1-Tenv_0)/(zenv_plus1-zenv_0)
        T=Tenv_0+(z-zenv_0)*dTdz
        j+=1
    return T

def wvenv(z):
    j=0
    if data_env[j,0] == z or data_env[j,0] > z: #to prevent it from leaving domain
        wv=data_env[0,2]
    elif data_env[-1,0] < z: #to prevent it from leaving the domain
        wv=0
    while data_env[j,0] < z:
        wvenv_plus1=data_env[(j+1),2]
        wvenv_0=data_env[j,2]
        zenv_plus1=data_env[(j+1),0]
        zenv_0=data_env[j,0]
        dwvdz=(wvenv_plus1-wvenv_0)/(zenv_plus1-zenv_0)
        wv=wvenv_0+(z-zenv_0)*dwvdz
        j+=1
    return wv

for t in t1:
    Tenv_new=Tenv(zp)
    wvenv_new=wvenv(zp)
    w_old=w
    zp_old=zp
    Tp_old=Tp
    wv_old=wvp
    w=w+dwdt(Tp,Tenv_new)*dt
    zp=zp+w_old*dt
    Tp=Tp+dTpdt(w_old,Tp_old,zp_old)*dt
    wvp=wvp+dwvdt(w_old,wv_old,wvenv_new)*dt
    data_p=np.append(data_p,np.array([zp,Tp,w,wvp,Tenv_new,wvenv_new]))

data_p=np.reshape(data_p,(int(len(data_p)/6.0),6))
pl.plot(data_p[:,4],data_p[:,0])
pl.plot(data_p[:,1],data_p[:,0])
pl.ylim(0,1.1*np.max(data_p[:,0]))
pl.show()

pl.plot(data_p[:,5],data_p[:,0])
pl.plot(data_p[:,3],data_p[:,0])

data=np.zeros((3,len(t1)+1))


for t in t1:
#    w=wnew(thetap,z,w)
 #   z=znew(thetap,z,w)
 #   print(w,z,t)   
    data[0,int(t/dt)]=z
    data[1,int(t/dt)]=w
    data[2,int(t/dt)]=t
    



