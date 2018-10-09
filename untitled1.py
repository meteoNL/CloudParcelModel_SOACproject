# -*- coding: utf-8 -*-
"""
Created on Fri Oct 05 08:12:20 2018

@author: Edward chenxi
"""
import numpy as np
import matplotlib.pyplot as pl

#constants
g=9.81 #gravitational acceleration
cp=1005. #specific heat per kilogram of dry air
T0=273.15 #zero Celsius Kelvin reference temperature
Rv=461.5 #gas constant water vapor
Rd=287.05 #gas constant dry air
Lv=2.2647e6 #latent heat of vaporization water
es0=610.78 #reference saturation vapor pressure
epsilon=0.622 #molar mass ratio water and dry air

#time space
tend=7200. #end of the simulation, s
dt=1. #time step, s
t1=np.linspace(0.0,tend,int(tend/dt)) 
#print (len(t1))

#parameters 
gamma=0.5 #induced relation with environmental air, inertial
mu=2e-4 #entrainment of air
tau_cond = 30. #time scale for condensation, s
tau_evap = 30. #time scale for evaporation, s
tau_warmpc = 20.*60 #time scale for the formation of warm precipitation, s
C_evap=1400.
#%%
#read background data from 20090526_00z_De_Bilt
data_env=np.array([])
f=open('20090526_00z_De_Bilt.txt','r')
p_d = np.array([])
z = np.array([])
T = np.array([])
wv = np.array([])
i=0
for line in f:
    line=line.split(';')
    p_d = np.append(p_d, float(line[1])*100.) #read pressure and convert to Pa
    z = np.append(z, float(line[2])) #read height in meters
    T = np.append(T, float(line[3])+T0) #read temperature and convert to Kelvin
    wv = np.append(wv, float(line[6])/1000.) #read water vapor mixing ratio and convert to kg/kg
    i+=1
f.close()
#arrays for data in the environment and in the parcel, p:parcel env:environment
sat = np.zeros(len(t1))
zp = np.zeros(len(t1))
Tp = np.zeros(len(t1))
w = np.zeros(len(t1))
wvp = np.zeros(len(t1))
wvenv = np.zeros(len(t1))
p = np.zeros(len(t1))
Tenv = np.zeros(len(t1))
wL = np.zeros(len(t1))
total_prec = np.zeros(len(t1))
sat = np.zeros(len(t1))
C = np.zeros(len(t1))
E = np.zeros(len(t1))
#%%
#interpolate T and wv profiles, linear interpolation y=a*x+b where a = d/dz of the respective variable and b is the reference value that was measured
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def Tenvcalc(h):
    i = find_nearest(z,h)
    if i == 0:
        Tenv = T[0]
    elif i == len(z):
        Tenv = 300
    else:
        dTdz=(T[i+1]-T[i-1])/(z[i+1]-z[i-1]) #gradient a
        Tenv = T[i]+(h-z[i])*dTdz
    return Tenv
def wvenvcalc(h):
     i = find_nearest(z,h)
     if i == 0:
        wvenv = wv[0]
     elif i == len(z):
        wvenv = 0
     else:
        dwvdz = (wv[i+1]-wv[i-1])/(z[i+1]-z[i-1])
        wvenv = wv[i]+(h-z[i])*dwvdz
     return wvenv
#%%
#initial conditions
Tp[0] = 288.5 #initial temperature of parcel, K
zp[0] = 1500. #initial height of parcel, m
w[0] = 0. #initial velocity of parcel, m/s
wvp[0] = 10.9/1000. #mixing ratio of water vapor of parcel, kg/kg
wL[0] = 0. #cloud content
total_prec[0] = 0.
p[0] = p_d[0]
#%%
#differential equations
def dwdt(w,Tp,Tenv,wL): 
    return 1/(1+gamma)*(g*((Tp-Tenv)/Tenv-wL)-mu*abs(w)*w)

def dTpdt(w,Tp,Tenv,zp,C,E):
    return -g*w/cp-mu*abs(w)*(Tp-Tenv)+Lv/cp*(C-E)

def dwvpdt(w,wvp,wvenv,C,E):
    return -mu*(wvp-wvenv)*abs(w)-C+E

def dpdt(rho,w):
    return -rho*g*w

def dwLdt(w,C,E,wL,warm_precip):
    return C-E-warm_precip-mu*wL*abs(w)

def func(phi,C,E,warm_precip,rho,Tenv,wvenv,t):#phi = [p,w,zp,Tp,wvp,wL]
    w=phi[1]
    zp=phi[2]
    Tp=phi[3]
    wvp=phi[4]
    wL=phi[5]
    dp=dpdt(rho,w)*dt
    dw=dwdt(w,Tp,Tenv,wL)*dt
    dzp=w*dt
    dTp=dTpdt(w,Tp,Tenv,zp,C,E)*dt
    dwvp=dwvpdt(w,wvp,wvenv,C,E)*dt
    dwL=dwLdt(w,C,E,wL,warm_precip)*dt
    return np.array([dp,dw,dzp,dTp,dwvp,dwL])
#%%
def wvscalc(T,p):#calculation of water vapor saturation mixing ratio
    #from Aarnouts lecture notes and Wallace and Hobbs
    diffT=(1/T0-1/T)
    difflnes=Lv/Rv*diffT
    lnes=difflnes+np.log(es0)
    es=np.exp(lnes)
    wv=epsilon*(es/(p-es))
    return wv

def condensation(wv,wvs):
    if wv > wvs:
        return (wv-wvs)*(1-np.exp(-1/tau_cond*dt))
    else:
        return 0.00

def evaporation(wv,wvs,wL):
    if wvs > wv and wL>0:
        return C_evap*wL*(wvs-wv)*((1-np.exp(-1/tau_evap*dt)))
    else:
        return 0.00
    
def warm_precip(wL):
    if wL > 0:
        return wL*(1-np.exp(-dt/tau_warmpc))
    else:
        return 0.0
#%%Integration
t=0
Tenv[0] = Tenvcalc(zp[0])
wvenv[0] = wvenvcalc(zp[0]) 
sat[0] = wvp[0]/wvscalc(Tp[0],p[0])
C[0] = condensation(wvp[0],wvscalc(Tp[0],p[0]))
E[0] = evaporation(wvp[0],wvscalc(Tp[0],p[0]),0)
for i in range(len(t1)-1): 
    #do the gass law and hydrostatic equilibrium to calculate pressure and saturation
    # !! HERE WE STILL USE EULER FORWARD, HOWEVER PRESSURE IS NOT AS IMPORTANT AS THE OTHERS BECAUSE IT IS ONLY APPROX. NEEDED FOR CONDENSATION !!
    Tv = Tp[i]*(1+(wvp[i])/epsilon)/(1+wvp[i]) #virtual temp, from Aarnouts lecture notes
    rho = p[i]/(Rd*Tv) #gas law
    #Runge- Kutta numerical scheme 
    phi=np.array([p[i],w[i],zp[i],Tp[i],wvp[i],wL[i]])
    k1,k2,k3,k4=np.zeros(6),np.zeros(6),np.zeros(6),np.zeros(6)
    k1[:]=func(phi, C[i],E[i],warm_precip(wL[i]),rho,Tenv[i], wvenv[i],t)
    k2[:]=func((phi+0.5*k1), C[i],E[i],warm_precip(wL[i]),rho,Tenv[i], wvenv[i],(t+0.5*dt))
    k3[:]=func((phi+0.5*k2), C[i],E[i],warm_precip(wL[i]),rho,Tenv[i], wvenv[i],(t+0.5*dt))
    k4[:]=func((phi+k3), C[i],E[i],warm_precip(wL[i]),rho,Tenv[i], wvenv[i],(t+dt))

    #update values and save them in resulting array that includes time
    phi=phi+np.array((1./6)*(k1+2*k2+2*k3+k4),dtype='float64')
    p[i+1]=phi[0]
    w[i+1]=phi[1]
    zp[i+1]=phi[2]
    Tp[i+1]=phi[3]
    wvp[i+1]=phi[4]  
    wL[i+1]=phi[5]
    total_prec[i+1]=total_prec[i]+warm_precip(wL[i+1]) #total precipitation
    Tenv[i+1] = Tenvcalc(zp[i+1])
    wvenv[i+1] = wvenvcalc(zp[i+1]) 
    wvs = wvscalc(Tp[i+1],p[i+1]) #water vapor saturation mixing ratio 
    sat[i+1] = wvp[i+1]/wvs
    C[i+1]=condensation(wvp[i+1],wvs)
    E[i+1]=evaporation(wvp[i+1],wvs,wL[i+1])
    warm_prec=warm_precip(wL[i+1])
    t = t + dt
#%%plot
pl.plot(Tp,zp,c='r',label='Tp')
pl.plot(Tenv,zp,c='g',label='Tenv')
pl.legend()
pl.ylim(0,14000)
pl.show()
pl.plot(t1,sat)
pl.show()
pl.plot(t1,zp)
pl.ylim(0,14000)    
pl.show()
pl.plot(t1,total_prec)
pl.show()

