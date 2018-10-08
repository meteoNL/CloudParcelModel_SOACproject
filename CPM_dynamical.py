# -*- coding: utf-8 -*-
"""
Created on Fri Oct 05 08:12:20 2018

@author: Edward chenxi
"""

import numpy as np
import matplotlib.pyplot as pl

#constants set for this simulation and initial conditions
tend=3600. #end of the simulation 
dt=1. #time step
gamma=0.5 #induced relation with environmental air, inertial
mu=2e-4 #entrainment of air

Tp=288.5 #initial temp air parcel, K
zp=1500. #initial height air parcel, m
w=0. #initial velocity air parcel, m/s
wvp=10.9/1000. #mixing ratio of water vapor kg/kg
wL = 0. #initial condition for cloud content

g=9.81 #gravitational acceleration
cp=1005. #specific heat per kilogram of air
T0=273.15 #zero Celsius Kelvin reference temperature
Rv=461.5 #gass constant water vapor
Rd=287.05 #gass constant dry air
Lv=2.5e6 #latent heat of vaporization water
es0=610.78 #reference saturation vapor pressure
epsilon=0.622 #molar mass ratio water and dry air
tau_cond = 30. #time scale for condensation
tau_evap = 10.*60. #time scale for evaporation
#our time space
t1=np.linspace(0.0,tend,int(tend/dt)) 

#some arrays for data in the environment and in the parcel
data_env=np.array([])
sat_arr=np.array([])
zp_arr = np.array([zp])
Tp_arr = np.array([Tp])
w_arr = np.array([w])
wvp_arr = np.array([wvp])
pp_arr = np.array([])
Tenvarr=np.array([])
wvenvarr=np.array([])
wL_arr=np.array([])
#%%
#read some environmental data, for now 28th of June 2011, Essen (Germany), 12 UTC
f=open('20090526_00z_De_Bilt.txt','r')
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
pp_arr=np.append(pp_arr,pp)
#reshape data enviromental air to matrix and close file 
data_env=np.reshape(data_env,(int(len(data_env)/4.0),4))
f.close()
#%%
#differential equations
def dwdt(Tp,Tenv,wL):
    return 1/(1+gamma)*(g*((Tp-Tenv)/Tenv-wL)-mu*abs(w)*w)

def dTpdt(w,Tp,zp,cond,evap):
    return -(g*w/cp+mu*abs(w)*(Tp-Tenv(zp)))+Lv/cp*(cond-evap)

def dwvdt(w,wvp,wvenv,cond,evap):
    return -mu*(wvp-wvenv)*abs(w)-(cond-evap)

def dpdt(rho,w):
    return (-rho*g*w)

def dwLdt(w,wL,cond,evap):
    return cond-evap-mu*wL*abs(w)

def func(phi,cond,evap,rho,Tenv,wvenv,t):
    pp=phi[0]
    w=phi[1]
    zp=phi[2]
    Tp=phi[3]
    wvp=phi[4]
    wL=phi[5]
    dp=dpdt(rho,w)*dt
    dw=dwdt(Tp,Tenv,wL)*dt
    dzp=w*dt
    dTp=dTpdt(w,Tp,zp,cond,evap)*dt
    dwvp=dwvdt(w,wvp,wvenv,cond,evap)*dt
    dwL=dwLdt(w,wL,cond,evap)*dt
    phi=np.array([dp,dw,dzp,dTp,dwvp,dwL])
    return phi


#interpolated temperature and water vapor profiles, linear interpolation y=a*x+b where a = d/dz of the respective variable and b is the reference value that was measured
def Tenv(z):
    j=0
    if data_env[j,0] == z or data_env[j,0] > z: #to prevent it from leaving domain
        T=data_env[0,1]
    elif data_env[-2,0] < z: #to prevent it from leaving the domain
        T=300
    while data_env[(j+1),0] < z:
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

def condensation(wv,wvs):
    if wv > wvs:
        return (wv-wvs)*(1-np.exp(-1/tau_cond*dt))
    else:
        return 0.00

def evaporation(wv,wvs,wL):
    if wvs > wv and wL>0:
        return wL*(wvs-wv)*((1-np.exp(-1/tau_evap*dt)))
    else:
        return 0.00

for t in t1:
    #calculate environmental values of water vapor and temperature
    Tenv_new=Tenv(zp)
    wvenv_new=wvenv(zp)
    
    #save the old values first
    w_old=w
    zp_old=zp
    Tp_old=Tp
    wv_old=wvp
    wL_old=wL
    
    #do the gass law and hydrostatic equilibrium to calculate pressure and saturation
    Tv=Tp_old*(1+(wv_old)/epsilon)/(1+wv_old) #Aarnouts lecture notes
    rho_old=pp/(Rd*Tv) #gas law
    wvs_old=wvscalc(Tp_old,pp)
    saturation=wv_old/wvs_old
    sat_arr=np.append(sat_arr,saturation)
    wvenvarr=np.append(wvenvarr,wvenv_new)
    Tenvarr=np.append(Tenvarr,Tenv_new)
    cond=condensation(wv_old,wvs_old)
    evap=evaporation(wv_old,wvs_old,wL)
    #do the differential equations
    phi=np.array([pp,w,zp,Tp,wvp,wL])
    k1,k2,k3,k4=np.zeros(6),np.zeros(6),np.zeros(6),np.zeros(6)
    k1[:]=func(phi,cond,evap,rho_old,Tenv_new,wvenv_new,t)
    k2[:]=func((phi+0.5*k1),cond,evap,rho_old,Tenv_new,wvenv_new,(t+0.5*dt))
    k3[:]=func((phi+0.5*k2),cond,evap,rho_old,Tenv_new,wvenv_new,(t+0.5*dt))
    k4[:]=func((phi+k3),cond,evap,rho_old,Tenv_new,wvenv_new,(t+dt))
    #update values and save them in resulting array that includes time
    phi=phi+np.array((1./6)*(k1+2*k2+2*k3+k4),dtype='float64')
    pp=phi[0]
    w=phi[1]
    zp=phi[2]
    Tp=phi[3]
    wvp=phi[4]  
    wL=phi[5]
    #add data to array
    zp_arr = np.append(zp_arr,zp)
    Tp_arr = np.append(Tp_arr,Tp)
    w_arr=np.append(w_arr,w)
    wvp_arr=np.append(wvp_arr,wvp)
    pp_arr=np.append(pp_arr,pp)
    wL_arr=np.append(wL_arr,wL)

#calculate environmental values of water vapor and temperature
Tenv_new=Tenv(zp)
wvenv_new=wvenv(zp)

#save the old values first
w_old=w
zp_old=zp
Tp_old=Tp
wv_old=wvp
wL_old=wL

#do the gass law and hydrostatic equilibrium to calculate pressure and saturation
Tv=Tp_old*(1+(wv_old)/epsilon)/(1+wv_old) #Aarnouts lecture notes
rho_old=pp/(Rd*Tv) #gas law
wvs_old=wvscalc(Tp_old,pp)
saturation=wv_old/wvs_old
sat_arr=np.append(sat_arr,saturation)
wvenvarr=np.append(wvenvarr,wvenv_new)
Tenvarr=np.append(Tenvarr,Tenv_new)
wL_arr=np.append(wL_arr,wL)
 
pl.plot(Tp_arr,zp_arr)
pl.plot(Tenvarr,zp_arr)
pl.ylim(0,14000)
pl.show()
pl.plot(t1,sat_arr[:-1])
pl.show()
pl.plot(t1,zp_arr[:-1])
    



