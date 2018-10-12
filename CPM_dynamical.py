# -*- coding: utf-8 -*-
"""
Created on Fri Oct 05 08:12:20 2018

@author: Edward and Chenxi
"""
import numpy as np
import matplotlib.pyplot as pl

#input of the model
#constants
g=9.81 #gravitational acceleration
cp=1005. #specific heat per kilogram of dry air
T0=273.15 #zero Celsius Kelvin reference temperature
Rv=461.5 #gas constant water vapor
Rd=287.05 #gas constant dry air
Lf = 3.35e5
es0=610.78 #reference saturation vapor pressure
T1=273.16
T2=235
es1=611.20
epsilon=0.622 #molar mass ratio water and dry air
Ka = 2.4e-2 #Thermal conductivity of air
rhoi = 700 #density of ice cristal, kg/m3

#pseudoconstants
def chi(p): #diffusivity of water vapor
    return 2.21/p
def A(T):
    return Ls(T)/Ka/T*(Ls(T)/(Rv*T)-1)
def Lv(T):#latent heat of vaporization water
    return (2.501 - 2.361e-3*(T-T0))*1e6
def Ls(T): #latent heat of sublimation water
    return Lf+ Lv(T)

#time space
tend=7200. #end of the simulation, s
dt=1. #time step, s
t1=np.linspace(0.0,tend,int(tend/dt)) 
dz=1.

#parameters 
gamma=0.5 #induced relation with environmental air, inertial
mu=0.5e-4 #entrainment of air: R.A. Anthes (1977) gives 0.183/radius as its value
tau_cond = 30. #time scale for condensation, s
tau_evap = 30. #time scale for evaporation, s
tau_warmpc = 90.*60 #time scale for the formation of warm precipitation, s, 1000 s in Anthes (1977); the idea appears to be from Kessler (1969)
tau_coldpc = 12.*60 #time scale for the formation of cold precipitation, 700 s in ECMWF doc mentioned
C_evap=1400.
wLthres=4.5e-4 # threshold for precip based on ECMWF documentation; 5e-4 in Anthes (1977)
withres=wLthres #threshold for precip form from ice
Cconv = 2.00 #assumed constant for increased rate in deposition in convective clouds compared to shallow stratiform clouds


#%%
#read background data from 20090526_00z_De_Bilt
fn='20090526_00z_De_Bilt.txt'
f=open('20090526_00z_De_Bilt.txt','r')
p_d = np.array([])
z = np.array([])
T = np.array([])
wv = np.array([])
Td=np.array([])
for line in f:
    line=line.split(';')
    p_d = np.append(p_d, float(line[1])*100.) #read pressure and convert to Pa
    z = np.append(z, float(line[2])) #read height in meters
    T = np.append(T, float(line[3])+T0) #read temperature and convert to Kelvin
    Td = np.append(Td,float(line[4])+T0)
    wv = np.append(wv, float(line[6])/1000.) #read water vapor mixing ratio and convert to kg/kg
f.close()

#%%
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
wi = np.zeros(len(t1))
total_prec = np.zeros(len(t1))
sat = np.zeros(len(t1))
C = np.zeros(len(t1))
E = np.zeros(len(t1))
total_water=np.zeros((len(t1)))

#%% envirmental profiles used
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
        if h == z[i]:
            Tenv = T[i]
        elif h > z[i]:
            dTdz=(T[i+1]-T[i])/(z[i+1]-z[i])
        else:
            dTdz=(T[i]-T[i-1])/(z[i]-z[i-1])  
        Tenv = T[i]+(h-z[i])*dTdz
    return Tenv
def wvenvcalc(h):
     i = find_nearest(z,h)
     if i == 0:
        wvenv = wv[0]
     elif i == len(z):
        wvenv = 0
     else:
        if h == z[i]:
            wvenv = wv[i]
        elif h > z[i]:
            dwvdz=(wv[i+1]-wv[i])/(z[i+1]-z[i])
        else:
            dwvdz=(wv[i]-wv[i-1])/(z[i]-z[i-1])  
        wvenv = wv[i]+(h-z[i])*dwvdz
     return wvenv
 
def p0(zloc,dz):
    #locate layer in which parcel is
    i=0
    while zloc > z[i+1]:
        i+=1
        
    #get properties at the base of this layer (lower bound, pressure & height) and layer means (temp & water vapor)    
    zval=z[i]
    pref=p_d[i]

    while zval < zloc:
        #integrate hydrostatic equilibrium with EF and given dz
        zval+=0.5*dz
        Tloc=Tenvcalc(zval)
        wvloc=wvenvcalc(zval)
        Tvloc=Tloc*(1+(wvloc)/epsilon)/(1+wvloc) #from Aarnouts lecture notes
        rho = pref/(Rd*Tvloc)
        dpdz=-rho*g
        pref+=dpdz*dz
        zval+=0.5*dz
    return pref

#%%
#initial conditions
Tp[0] = 288.5 #initial temperature of parcel, K
zp[0] = 1500. #initial height of parcel, m
w[0] = 0. #initial velocity of parcel, m/s
wvp[0] = 10.9/1000. #mixing ratio of water vapor of parcel, kg/kg
wL[0] = 0. #cloud content
total_prec[0] = 0.
p[0] = p0(zp[0],dz)

#%%
#differential equations
def dwdt(w,Tp,Tenv,wL): 
    return 1./(1.+gamma)*(g*((Tp-Tenv)/Tenv-wL-wi[i])-mu*abs(w)*w)

def dTpdt(w,Tp,Tenv,zp,C,E,dwi):
    return -g*w/cp-mu*abs(w)*(Tp-Tenv)+Lv(Tp)/cp*(C-E)+dwi*Lf/cp

def dwvpdt(w,wvp,wvenv,C,E):
    return -mu*(wvp-wvenv)*abs(w)-C+E

def dpdt(rho,w):
    return -rho*g*w

def dwLdt(w,C,E,wL,warm_precip,dwi):
    return C-E-warm_precip-mu*wL*abs(w)-dwi

def func(phi,procarg,rho):#C,E,warm_precip,rho,Tenv,wvenv,t):#phi = [p,w,zp,Tp,wvp,wL]
    #extract values
    w,zp,Tp,wvp,wL=phi[1],phi[2],phi[3],phi[4],phi[5]
    C,E,warm_precip,Tenv,wvenv,dwi=procarg[0],procarg[1],procarg[2],procarg[3],procarg[4],procarg[5]

    #do the diff eqs
    dp=dpdt(rho,w)*dt
    dw=dwdt(w,Tp,Tenv,wL)*dt
    dzp=w*dt
    dTp=dTpdt(w,Tp,Tenv,zp,C,E,dwi)*dt
    dwvp=dwvpdt(w,wvp,wvenv,C,E)*dt
    dwL=dwLdt(w,C,E,wL,warm_precip,dwi)*dt
    return np.array([dp,dw,dzp,dTp,dwvp,dwL])

#%% thermodynamic equilibria over water/ice surfaces
def escalc(T):
    #from Aarnouts lecture notes and Wallace and Hobbs
    diffT=(1./T0-1./T)
    difflnes=Lv(T)/Rv*diffT
    lnes=difflnes+np.log(es0)
    es=np.exp(lnes)
    return es

def wvscalc(T,p):#calculation of water vapor saturation mixing ratio
    #from Aarnouts lecture notes and Wallace and Hobbs
    es=escalc(T)
    wvsat=epsilon*(es/(p-es))
    return wvsat

def esicalc(T,p):#calculation of water vapor saturation mixing ratio
    #fromUniversity of North Carolina lecuture slides
    diffT=(1./T1-1./T)
    difflnesi=Ls(T)/Rv*diffT
    lnesi=difflnesi+np.log(es1)
    esi = np.exp(lnesi)
    return esi

#%%
#processes: phase changes
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

#deposition of cloud water to solid phase
def B(T,p):
    return Rv*T*chi(p)*esicalc(T,p)    
def Ni(T,p):
    return 1e3*np.exp(12.96*(escalc(T)-esicalc(T,p))/esicalc(T,p)-0.639)
def cvd(T,p,rho):
    return Cconv*7.8*(Ni(T,p)**(2./3)*(escalc(T)-esicalc(T,p)))/(rho**(1./3)*(A(T)+B(T,p))*esicalc(T,p))
def Wi_depmeltfreez(T,p,rho,wL,t):
    if T > T2 and T < T0:
        result=(2./3*cvd(T,p,rho)*t+wi[(i-1)]**(2./3))**(3./2)
        if result < wL:
            return result
        else:
            return wi[i]+wL
    elif T < T2:
        return wi[i]+wL
    else:
        return 0.00

#%% precipitation processes
#warm precipitation (mainly autoconversion simulation)
def warm_precip(wL,Tp):
    if wL > wLthres and Tp > T0:
        return (wL-wLthres)*(1-np.exp(-dt/tau_warmpc))
    else:
        return 0.0

#cold precipitation
def cold_precip(wL,wi):
    result1=(wi-withres)*(1-np.exp(-dt/tau_coldpc))
    if wi > withres:
        return result1
    else:
        return 0.00
    return 0.00
    
#%%Integration procedure
t=t1[0]
Tenv[0] = Tenvcalc(zp[0])
wvenv[0] = wvenvcalc(zp[0]) 
sat[0] = wvp[0]/wvscalc(Tp[0],p[0])
C[0] = condensation(wvp[0],wvscalc(Tp[0],p[0]))
E[0] = evaporation(wvp[0],wvscalc(Tp[0],p[0]),0)
dwi=0.
for i in range(len(t1)-1): 
    #do the gass law and hydrostatic equilibrium to calculate pressure and saturation
    # !! HERE WE STILL USE EULER FORWARD, HOWEVER PRESSURE IS NOT AS IMPORTANT AS THE OTHERS BECAUSE IT IS ONLY APPROX. NEEDED FOR CONDENSATION !!
    Tv = Tp[i]*(1+(wvp[i])/epsilon)/(1+wvp[i]) #virtual temp, from Aarnouts lecture notes
    rho = p[i]/(Rd*Tv) #gas law
    #Runge- Kutta numerical scheme 
    processargs=np.array([C[i],E[i],warm_precip(wL[i],Tp[i]),Tenv[i],wvenv[i],dwi])
    phi=np.array([p[i],w[i],zp[i],Tp[i],wvp[i],wL[i]])
    k1,k2,k3,k4=np.zeros(6),np.zeros(6),np.zeros(6),np.zeros(6)
    k1[:]=func(phi, processargs,rho)
    k2[:]=func((phi+0.5*k1), processargs,rho)
    k3[:]=func((phi+0.5*k2), processargs,rho)
    k4[:]=func((phi+k3), processargs,rho)

    #update values and save them in resulting array that includes time
    phi=phi+np.array((1./6)*(k1+2*k2+2*k3+k4),dtype='float64')
    t=t1[i+1]
    p[i+1]=phi[0]
    w[i+1]=phi[1]
    zp[i+1]=phi[2]
    Tp[i+1]=phi[3]
    wvp[i+1]=phi[4]  
    wL[i+1]=phi[5]
    
    #update parcel environment
    Tenv[i+1] = Tenvcalc(zp[i+1])
    wvenv[i+1] = wvenvcalc(zp[i+1]) 
    
    #calculate saturation values
    wvs = wvscalc(Tp[i+1],p[i+1]) #water vapor saturation mixing ratio 
    sat[i+1] = wvp[i+1]/wvs    
    
    #then do condencsation, evaporation, freezing, melting, deposition/Findeisen-Wegener-Bergeron process
    C[i+1]=condensation(wvp[i+1],wvs)
    E[i+1]=evaporation(wvp[i+1],wvs,wL[i+1])
    wi[(i+1)]=Wi_depmeltfreez(Tp[i+1],p[i+1],rho,wL[i+1],dt)
    dwi=wi[i+1]-wi[i]
    
    #precipitation process of the clouds and remove the cold precip from ice parcels
    warm_prec=warm_precip(wL[i+1],Tp[i+1])
    cold_prec=cold_precip(wL[i+1],wi[i+1])
    wi[i+1]=wi[i+1]-cold_prec
    total_prec[i+1]=total_prec[i]+warm_prec+cold_prec #update total precipitation
    
#%% visualization of results
#plot temerature profile
gamma=0.0050 #skew T visualzation constant
xticks=np.array([])
z_plot=np.arange(0,15000,1000)
for i in range(193,310,5):
    pl.plot(i*np.ones(len(z_plot))+gamma*z_plot,z_plot,c=(0.6,0.6,0.6))
    if i > 265 and i < 300:
        xticks=np.append(xticks,np.array([i]))
pl.plot((Tp+gamma*zp),zp,c='r',label='Tp')
pl.plot((Td+gamma*z),z,c='b',label='Tdew',ls='--')
pl.plot((T+gamma*z),z,c='g',label='Tenv')
pl.title(fn[:-4])
pl.xlim(265,300)
pl.xticks(xticks,(xticks-273))
pl.legend(loc=1)
pl.ylim(0,14000)
pl.xlabel('Temperature (degrees Celsius)')
pl.ylabel('Height (m)')
pl.show()

#saturation evolution plot
pl.plot(t1,sat)
pl.show()

#height evolution of parcel
pl.plot(t1,zp)
pl.ylim(0,14000)    
pl.show()

#rain event evolution
pl.plot(t1,total_prec)
pl.show()

#cloud composition as function of temperature
pl.plot(wL,Tp,label='Cloud liquid water')
pl.plot(wi,Tp,label='Cloud ice')
pl.legend()