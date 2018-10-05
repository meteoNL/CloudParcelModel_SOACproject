# -*- coding: utf-8 -*-
"""
Created on Fri Oct 05 08:12:20 2018

@author: Edward chenxi
"""

import numpy as np
import matplotlib.pyplot as pl

#constants set for this simulation and initial conditions
tend=3600. #end of the simulation 
dt=0.5 #time step
gamma=0.5 #induced relation with environmental air, inertial
mu=5e-4 #entrainment of air

Tp=308. #initial temp air parcel, K
zp=170. #initial height air parcel, m
w=0. #initial velocity air parcel, m/s
wvp=15.0/1000. #mixing ratio of water vapor kg/kg

g=9.81 #gravitational acceleration
cp=1005. #specific heat per kilogram of air
T0=273.15 #zero Celsius Kelvin reference temperature
Rv=461.5 #gass constant water vapor
Rd=287.05 #gass constant dry air
Lv=2.5e6 #latent heat of vaporization water
es0=610.78 #reference saturation vapor pressure
epsilon=0.622 #molar mass ratio water and dry air

#our time space
t1=np.linspace(0.0,tend,int(tend/dt)) 

#some arrays for data in the environment and in the parcel
data_env=np.array([])
data_p=np.array([])
#%%
#read some environmental data, for now 28th of June 2011, Essen (Germany), 12 UTC
f=open('dataset.txt','r')
i=0
for line in f:
    line=line.split(';')
    p=float(line[1])*100. #read pressure and convert to Pa
    z=float(line[2]) #read height in meters
    T=float(line[3])+T0 #read temperature and convert to Kelvin
    wv=float(line[6])/1000. #read water vapor mixing ratio and convert to kg/kg
    data_env=np.append(data_env,np.array([z,T,wv,p])) #put data in array
    i+=1

pp=float(data_env[3])  #initial pressure condition (of the environmental air)
    
#reshape data enviromental air to matrix and close file 
data_env=np.reshape(data_env,(int(len(data_env)/4.0),4))
f.close()
#%%
#differential equations
def dwdt(Tp,Tenv):
    return 1/(1+gamma)*(g*(Tp-Tenv)/Tenv-mu*abs(w)*w)

def dTpdt(w,Tp,zp):
    return -(g*w/cp+mu*abs(w)*(Tp-Tenv(zp)))

def dwvdt(w,wvp,wvenv):
    return -mu*(wvp-wvenv)*abs(w)

def dpdt(rho,w):
    return (-rho*g*w)

#interpolated temperature and water vapor profiles, linear interpolation y=a*x+b where a = d/dz of the respective variable and b is the reference value that was measured
def Tenv(z):
    j=0
    if data_env[j,0] == z or data_env[j,0] > z: #to prevent it from leaving domain
        T=data_env[0,1]
    elif data_env[-1,0] < z: #to prevent it from leaving the domain
        T=300
    while data_env[j,0] < z:
        Tenv_plus1=data_env[(j+1),1] #next value
        Tenv_0=data_env[j,1] #reference value
        zenv_plus1=data_env[(j+1),0] #next height value
        zenv_0=data_env[j,0] #reference height value
        dTdz=(Tenv_plus1-Tenv_0)/(zenv_plus1-zenv_0) #gradient a
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
        wvenv_plus1=data_env[(j+1),2] #next value
        wvenv_0=data_env[j,2] #reference value
        zenv_plus1=data_env[(j+1),0] #next height value
        zenv_0=data_env[j,0] #reference height value
        dwvdz=(wvenv_plus1-wvenv_0)/(zenv_plus1-zenv_0) #gradient a
        wv=wvenv_0+(z-zenv_0)*dwvdz
        j+=1
    return wv

#calculation of water vapor saturation mixing ratio
def wvscalc(T,p):
    #from Aarnouts lecture notes and Wallace and Hobbs
    diffT=(1/T0-1/T)
    difflnes=Lv/Rv*diffT
    lnes=difflnes+np.log(es0)
    es=np.exp(lnes)
    ws=epsilon*(es/(p-es))
    return ws

#initialize water vapor saturation mixing ratio
wvs_old=wvscalc(Tp,pp)

for t in t1:
    #calculate environmental values of water vapor and temperature
    Tenv_new=Tenv(zp)
    wvenv_new=wvenv(zp)
    
    #save the old values first
    w_old=w
    zp_old=zp
    Tp_old=Tp
    wv_old=wvp
    
    #do the gass law and hydrostatic equilibrium to calculate pressure
    Tv_old=Tp_old*(1+(wv_old)/epsilon)/(1+wv_old) #Aarnouts lecture notes
    rho_old=pp/(Rd*Tv_old) #gas law
    pp_old=pp
    wvs=wvs_old
    wvs_old=wvscalc(Tp_old,pp_old)
    saturation=wv_old/wvs_old
    
    #do the differential equations
    pp=pp+dpdt(rho_old,w_old)*dt
    w=w+dwdt(Tp,Tenv_new)*dt
    zp=zp+w_old*dt
    Tp=Tp+dTpdt(w_old,Tp_old,zp_old)*dt
    wvp=wvp+dwvdt(w_old,wv_old,wvenv_new)*dt
    data_p=np.append(data_p,np.array([zp,Tp,w,wvp,Tenv_new,wvenv_new,saturation,pp]))

data_p=np.reshape(data_p,(int(len(data_p)/8.0),8))
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
    



