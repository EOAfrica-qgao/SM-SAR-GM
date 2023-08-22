# -*- coding: utf-8 -*-
"""
Created on Tue Jun 06 13:34:39 2017

@author: Qi
"""

import glob
import numpy as np
from datetime import datetime
from datetime import timedelta
from matplotlib import pyplot as plt
import pandas as pd
import os
import gdal
import scipy as sp
from scipy.interpolate import interp1d  
    
p_size=str('100m')

track=str('133')
hh=19
mm=18



def db2linear(data):
    #data = filters.median_filter(data, size=7)
    data = 10**(data/10)
    return data
def linear2db(data):
    #data = filters.median_filter(data, size=7)
    data = 10*np.log10(data)
    return data
    
field_mask=("D:/01_Projects/01_EO_Africa/field.tif")
dataset = gdal.Open(field_mask)
geotransform = dataset.GetGeoTransform()
field = dataset.GetRasterBand(1).ReadAsArray()
# projection = dataset.GetProjection()
dataset = None
    
    
field[field == 0] = np.nan
mask_field=np.zeros((200,200))*np.nan
for x in range(200):
    for y in range(200):
        xl=x*10
        xr=(x+1)*10
        yb=y*10
        yu=(y+1)*10
        
        mask_field[x,y]=np.nanmean(field[xl:xr,yb:yu])


ii,jj=np.where(mask_field == 1)
i_min=min(ii)
i_max=max(ii)
j_min=min(jj)
j_max=max(jj)

vv_files = glob.glob("D:/01_Projects/01_EO_Africa/Data/2020_2021/S1/VV/*.tif")
vv_files.sort()
n=len(vv_files)

vh_files = glob.glob("D:/01_Projects/01_EO_Africa/Data/2020_2021/S1/VH/*.tif")
vh_files.sort()

ndvi = np.load('D:/01_Projects/01_EO_Africa/Data/2020_2021/npy/s2_ndvi_interp_daily.npy')
date_ndvi=np.load('D:/01_Projects/01_EO_Africa/Data/2020_2021/npy/date_list_daily.npy')
s2_ndvi=np.zeros((n))*np.nan
date_bc=np.empty((n,0)).tolist()
db_VV=np.zeros((n))*np.nan
db_VH=np.zeros((n))*np.nan
for i in range(n):
    vv_file= vv_files[i]
    date_str = os.path.basename(vv_file).split("_")[4].split('T')[0]
    ymd=list(date_str)
    y=("%s%s%s%s"%(ymd[0],ymd[1],ymd[2],ymd[3]))
    m=("%s%s"%(ymd[4],ymd[5]))
    d=("%s%s"%(ymd[6],ymd[7]))
    y=int(y)
    m=int(m)
    d=int(d)
    date_bc[i]=datetime(y,m,d)
    dataset = gdal.Open(vv_file)
    geotransform = dataset.GetGeoTransform()
    bc_VV = dataset.GetRasterBand(1).ReadAsArray()
    dataset = None
    bc_VV = db2linear(bc_VV)
    
    vh_file= vh_files[i]
    dataset = gdal.Open(vh_file)
    geotransform = dataset.GetGeoTransform()
    bc_VH = dataset.GetRasterBand(1).ReadAsArray()
    dataset = None
    bc_VH = db2linear(bc_VH)
    
    for x in range(200):
        for y in range(200):
            if mask_field[x,y]==1: 
                idx=np.where(date_ndvi==datetime.toordinal(date_bc[i]))[0][0]
                s2_ndvi[i]= ndvi[x,y,idx]
                
                xl=x*10
                xr=(x+1)*10
                yb=y*10
                yu=(y+1)*10
                
                vv_win=bc_VV[xl:xr,yb:yu]
                db_VV[i]=np.mean(vv_win)
                db_VV[i]=linear2db(db_VV[i])
                
                vh_win=bc_VH[xl:xr,yb:yu]
                db_VH[i]=np.mean(vh_win)
                db_VH[i]=linear2db(db_VH[i])

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ln1 = ax1.plot(date_bc, s2_ndvi,'g',label='NDVI')
ln2 = ax2.plot(date_bc, db_VV,'r',label='VV backscatter')
ln3 = ax2.plot(date_bc, db_VH,'b',label='VH backscatter')
# ln1 = ax1.plot(date_bc, s2_ndvi,'go',label='NDVI')
# ln2 = ax2.plot(date_bc, db_VV,'r+',label='VV backscatter')
# ln3 = ax2.plot(date_bc, db_VH,'b*',label='VH backscatter')

ax1.set_xlabel('Date')
ax1.set_ylabel('NDVI',fontsize=20)
ax2.set_ylabel('Backscatter',fontsize=20)
ax1.xaxis.set_tick_params(labelsize=15, rotation=45)
ax1.yaxis.set_tick_params(labelsize=15)
ax2.yaxis.set_tick_params(labelsize=15)
lns = ln1+ln2+ln3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0,fontsize=20)
plt.xlim(date_bc[0], date_bc[-1])  
plt.title('Target Field',fontsize=20)
plt.show()
# plt.savefig('D:/01_Projects/01_EO_Africa/Data/2020_2021/figures/NDVI_field.png')