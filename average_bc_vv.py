# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 12:21:59 2022

@author: qgao
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


def db2linear(data):
    #data = filters.median_filter(data, size=7)
    data = 10**(data/10)
    return data
def linear2db(data):
    #data = filters.median_filter(data, size=7)
    data = 10*np.log10(data)
    return data


bc_files = glob.glob("D:/01_Projects/01_EO_Africa/Data/2020_2021/S1/*.tif")
bc_files.sort()
n_bc = len(bc_files)
date_bc=np.empty((n_bc,0)).tolist()
bc_matrix=np.zeros((2001,2001, n_bc))*np.nan

for j in range(n_bc):
    bc_file= bc_files[j]
    date_str = os.path.basename(bc_file).split("_")[4].split('T')[0]
    ymd=list(date_str)
    y=("%s%s%s%s"%(ymd[0],ymd[1],ymd[2],ymd[3]))
    m=("%s%s"%(ymd[4],ymd[5]))
    d=("%s%s"%(ymd[6],ymd[7]))
    y=int(y)
    m=int(m)
    d=int(d)
    date_bc[j]=datetime(y,m,d)
    
    dataset = gdal.Open(bc_file)
    geotransform = dataset.GetGeoTransform()
    bc = dataset.GetRasterBand(1).ReadAsArray()
    projection = dataset.GetProjection()
    dataset = None
    bc = db2linear(bc)
    
    bc_matrix[:,:,j]= bc
   
    print (j)

np.save('D:/01_Projects/01_EO_Africa/Data/2020_2021/npy/bc_matrix_linear_VV.npy',bc_matrix)
np.save('D:/01_Projects/01_EO_Africa/Data/2020_2021/npy/bc_date_list_daily.npy',date_bc)

#%%
bc_matrix=np.load('D:/01_Projects/01_EO_Africa/Data/2020_2021/npy/bc_matrix_linear_VV.npy')
av_bc=np.zeros((2001,2001))*np.nan
av_bc=bc_matrix.mean(2)

av_bc=linear2db(av_bc)
plt.figure()
plt.imshow(av_bc,cmap='jet_r')
plt.colorbar()
plt.title('Mean of SAR Backscatter')
plt.savefig("D:/01_Projects/01_EO_Africa/Data/2020_2021/figures/Mean_VV.png",dpi=1500)

np.save('D:/01_Projects/01_EO_Africa/Data/2020_2021/npy/bc_mean_VV.npy',av_bc)

