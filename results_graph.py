# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 12:07:08 2018

@author: Chenxi, Edward
"""

import numpy as np
import matplotlib.pyplot as pl
radii=np.array([10.,100.,250.,500.,750.,1000.,1250.,1500.,2000.,2500.,3300.,4000.,5000.])
precip=np.array([0.,0.,0.01,0.04,0.11,0.26,0.43,1.20,2.62,4.99,6.58,6.75,9.22])
temp=np.array([290.,282.,280.,277.,276.,275.,272.,269.,265.,260.,258.,247.,242.])
fig, ax1 = pl.subplots(figsize=(12,8))
ax1.bar(np.log(radii),precip)#,linewidth=3.0)
ax2=ax1.twinx()
ax2.scatter(np.log(radii),temp,color='orange')
pl.xticks(np.log([10.,30.,100.,300.,1000.,3000.]),np.round(np.array([10.,30.,100.,300.,1000.,3000.])))
pl.xlim(2,8.3)
pl.title(r'Equilibrium $T_{p}$ and precipitation as function of $R_{eq}$')
ax1.set_xlabel(r'Initial parcel $R_{eq}$ (m)')
ax1.set_ylabel(r'Temperature (K)',color='orange')
ax2.set_ylabel(r'Precipitation (mm) ',color='b')
pl.show()