# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 14:14:32 2022

@author: qgao
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import glob
import os
import gdal
from datetime import datetime, timedelta
import pandas as pd
#==============================================================================
# 
# def integ(x, tck, constant=-1):
#     x = np.atleast_1d(x)
#     out = np.zeros(x.shape, dtype=x.dtype)
#     for n in range(len(out)):
#         out[n] = interpolate.splint(0, x[n], tck)
#         out += constant
#     return out
# 
# yint = integ(xnew, tck)
# plt.figure()
# plt.plot(xnew, yint, xnew, -np.cos(xnew), '--')
# plt.legend(['Cubic Spline', 'True'])
#==============================================================================


import numpy as np
import scipy as sp
from scipy.interpolate import interp1d


p_size=str('100m')
pol=str('VV')
track=str('133')
hh=19
mm=18

s2_files = glob.glob("D:/01_Projects/01_EO_Africa/Data/S2/*.tif")
s2_files.sort()
n_s2=len(s2_files)
s2_ndvi=np.zeros((200,200,n_s2))*np.nan
n_days=(datetime(2022,12,31).date()-datetime(2016,1,1).date()).days
s2_ndvi_interp=np.zeros((200,200,n_days))*np.nan
date_s2=np.empty((n_s2,0)).tolist()
# =============================================================================
# date_list = [datetime(2020,1,1).date() + timedelta(days=x) for x in range(n_days)]
# =============================================================================
date_list = [datetime.toordinal(datetime(2016,1,1).date()) + x for x in range(n_days)]

for i in range(n_s2):
    
    # read the first bc file
    s2_file = s2_files[i]
    dataset = gdal.Open(s2_file)
    geotransform = dataset.GetGeoTransform()
    s2 = dataset.GetRasterBand(1).ReadAsArray()
    dataset = None
    
    date_s2_str = os.path.basename(s2_file).split("_")[0].split("T")[0]
    
    date_s2_list=list(date_s2_str)
    y=("%s%s%s%s"%(date_s2_list[0],date_s2_list[1],date_s2_list[2],date_s2_list[3]))
    m=("%s%s"%(date_s2_list[4],date_s2_list[5]))
    d=("%s%s"%(date_s2_list[6],date_s2_list[7]))
    y=int(y)
    m=int(m)
    d=int(d)
    date_s2[i]=datetime(y,m,d).date()
    date_s2[i]=datetime.toordinal(date_s2[i])
    
    for x in range(200):
        for y in range(200):
            xl=x*10
            xr=(x+1)*10
            yb=y*10
            yu=(y+1)*10
            
            ndvi_win=s2[xl:xr,yb:yu]
            av_ndvi=np.mean(ndvi_win)
            if ~np.isnan(av_ndvi) and av_ndvi>0:
                s2_ndvi[x,y,i]=av_ndvi
    
for i in range(200):
    for j in range(200):
        x1=[]
        y1=[]
        for k in range(n_s2):
            if not np.isnan(s2_ndvi[i,j,k]):
                x1=np.append(x1,date_s2[k])
                y1=np.append(y1,s2_ndvi[i,j,k])
                
        ind_begin=date_list.index(x1[0])
        ind_end=date_list.index(x1[len(x1)-1])

        if len(x1)>3:
            new_x = date_list[ind_begin:ind_end]
            new_y = sp.interpolate.interp1d(x1, y1)(new_x)
            #new_y = sp.interpolate.interp1d(x1, y1, kind='cubic')(new_x)
            s2_ndvi_interp[i,j,ind_begin:ind_end]=new_y
np.save('D:/01_Projects/01_EO_Africa/Data/npy/s2_ndvi_interp_daily.npy',s2_ndvi_interp)
np.save('D:/01_Projects/01_EO_Africa/Data/npy/date_list_daily.npy',date_list)
