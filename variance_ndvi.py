# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 16:41:39 2022

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


import numpy as np
import scipy as sp
from scipy.interpolate import interp1d

ndvi_matrix=np.load('D:/01_Projects/01_EO_Africa/Data/2020_2021/npy/ndvi_interp_daily_full_resolution.npy')
var_ndvi=np.zeros((2001,2001))*np.nan
var_ndvi = np.nanvar(ndvi_matrix, axis=2)

plt.figure()
plt.imshow(var_ndvi,cmap='jet_r', vmin=0, vmax=0.03)
plt.colorbar()
plt.title('Variance of NDVI')
plt.savefig("D:/01_Projects/01_EO_Africa/Data/2020_2021/figures/Variance_NDVI.png",dpi=1500)

np.save('D:/01_Projects/01_EO_Africa/Data/2020_2021/npy/variance_ndvi.npy',var_ndvi)

#%%

mean_ndvi=np.zeros((2001,2001))*np.nan
mean_ndvi = np.nanmean(ndvi_matrix, axis=2)

plt.figure()
plt.imshow(mean_ndvi,cmap='jet_r', vmin=0, vmax=0.4)
plt.colorbar()
plt.title('Mean Value of NDVI')
plt.savefig("D:/01_Projects/01_EO_Africa/Data/2020_2021/figures/Mean_Value_NDVI.png",dpi=1500)

np.save('D:/01_Projects/01_EO_Africa/Data/2020_2021/npy/mean_ndvi.npy',mean_ndvi)

#%%
ndvi_matrix=np.load('D:/01_Projects/01_EO_Africa/Data/2020_2021/npy/ndvi_interp_daily_full_resolution.npy')
date_ndvi=np.load('D:/01_Projects/01_EO_Africa/Data/2020_2021/npy/date_list_daily.npy')
mean_ndvi_winter=np.zeros((2001,2001))*np.nan
mean_ndvi_winter = np.nanmean(ndvi_matrix[:,:,347:392], axis=2)

plt.figure()
plt.imshow(ndvi_matrix[:,:,347],cmap='jet_r', vmin=0, vmax=0.5)
plt.colorbar()
plt.title('Mean Value of NDVI in Winter Time')
plt.savefig("D:/01_Projects/01_EO_Africa/Data/2020_2021/figures/Mean_Value_NDVI_Winter.png",dpi=1500)

np.save('D:/01_Projects/01_EO_Africa/Data/2020_2021/npy/mean_ndvi_winter.npy',mean_ndvi_winter)
