# -*- coding: utf-8 -*-
"""
Created on Fri Oct 05 08:12:20 2018

@author: Edward
"""

import numpy as np
import matplotlib.pyplot as pl
g=9.81
tend=7200
dt=0.01
t1=np.linspace(1,tend,(tend/dt))

Tp=301 #temp air parcel, K
z=0
w=0

data_env=np.array([])

f=open('dataset.txt','r')
i=0
for line in f:
    line=line.split(';')
    print(line[2])
    z=float(line[2])
    T=float(line[3])
    data_env=np.append(data_env,np.array([z,T]))
    i+=1
data_env=np.reshape(data_env,(int(len(data_env)/2.0),2))


data=np.zeros((3,len(t1)+1))


for t in t1:
#    w=wnew(thetap,z,w)
 #   z=znew(thetap,z,w)
 #   print(w,z,t)   
    data[0,int(t/dt)]=z
    data[1,int(t/dt)]=w
    data[2,int(t/dt)]=t
    



