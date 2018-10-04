# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 15:14:00 2017

@author: weert
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 15:27:57 2017

@author: acer
"""
import numpy as np
import matplotlib.pyplot as pl
g=9.81
tend=2000
dt=0.01
t1=np.linspace(1,tend,(tend/dt))

theta0=300 #pot. temp environment at surface
thetap=301 #pot. temp air parcel
gamma1=-0.04 #temperature gradient
gamma2=0
gamma3=0.04
gamma4=0.005
z12=50 #height in meters
z23=2000
z34=2250

z=0
w=0

data=np.zeros((3,len(t1)+1))

def theta_z(z):
    if z< z12: 
        result=theta0+gamma1*z
    elif z<z23:
        result=theta0+gamma1*z12+gamma2*(z-z12)
    elif z<z34:
        result=theta0+gamma1*z12+gamma2*(z23-z12)+gamma3*(z-z23)
    else:
        result=theta0+gamma1*z12+gamma2*(z23-z12)+gamma3*(z34-z23)+gamma4*(z-z34)
    return result 

def a(thetap,z): #acceleration in w direction
    thetaz=theta_z(z)
    return g*(thetap-thetaz)/(thetaz)

def wnew(thetap,z,w):
    return w+a(thetap,z)*dt

def znew(thetap,z,w):
    return z+wnew(thetap,z,w)*dt

for t in t1:
    w=wnew(thetap,z,w)
    z=znew(thetap,z,w)
 #   print(w,z,t)   
    data[0,int(t/dt)]=z
    data[1,int(t/dt)]=w
    data[2,int(t/dt)]=t
    
pl.plot(data[2,:],data[0,:],color='b')
pl.ylabel('height (m)')
pl.xlabel('time (s)')
pl.title('B-V frequency')
pl.legend()
pl.grid()

vel=data[1,:]
maxvel=np.max(vel)
#print('maximum velocity = %s m/s' % round((maxvel),0))
#for i in range(len(vel)):
 #   if vel[i]==maxvel:
  #      j=i
   #     print('location of maximum velocity = %s m' % int(data[0,j]+0.5))
        
height=data[0,:]
maxheight=np.max(height)
print(maxheight)

print(str(9.81*(3/298*1950))+' '+str(9.81*2/299*50)+str(' ')+str(9.81*1.5/299.5*75))



#pl.plot(data[1,:],data[0,:])






    





