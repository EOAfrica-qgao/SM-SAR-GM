# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 14:34:00 2022

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


ndvi_files = glob.glob("D:/01_Projects/01_EO_Africa/Data/2020_2021/S2/*.tif")
ndvi_files.sort()
n_ndvi = len(ndvi_files)
date_ndvi=np.empty((n_ndvi,0)).tolist()
ndvi_matrix=np.zeros((2001,2001, n_ndvi))*np.nan

for j in range(n_ndvi):
    ndvi_file= ndvi_files[j]
    
    
    dataset = gdal.Open(ndvi_file)
    geotransform = dataset.GetGeoTransform()
    ndvi = dataset.GetRasterBand(1).ReadAsArray()
    projection = dataset.GetProjection()
    dataset = None
    
    ndvi_matrix[:,:,j]= ndvi
   
    print (j)

np.save('D:/01_Projects/01_EO_Africa/Data/2020_2021/npy/ndvi_matrix.npy',ndvi_matrix)

#%%
ndvi_matrix=np.load('D:/01_Projects/01_EO_Africa/Data/2020_2021/npy/ndvi_matrix.npy')
av_ndvi=np.zeros((2001,2001))*np.nan
av_ndvi=np.nanmean(ndvi_matrix, axis=2)

plt.figure()
plt.imshow(av_ndvi,cmap='jet_r')
plt.colorbar()
plt.title('Mean of SAR Backscatter')
plt.savefig("D:/01_Projects/01_EO_Africa/Data/2020_2021/figures/Mean_NDVI.png",dpi=1500)

np.save('D:/01_Projects/01_EO_Africa/Data/2020_2021/npy/ndvi_mean.npy',av_ndvi)


