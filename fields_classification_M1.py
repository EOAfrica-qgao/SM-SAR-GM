# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 16:56:16 2022

@author: qgao
"""

from matplotlib import pyplot as plt
import numpy as np
import gdal
import os

mask_out=("C:/Users/Qi/Dropbox/Toulouse2017/fields/tiff/mask_out.tif")
dataset = gdal.Open(mask_out)
geotransform = dataset.GetGeoTransform()
mask = dataset.GetRasterBand(1).ReadAsArray()
projection = dataset.GetProjection()
dataset = None


mean_map_VV=np.load('C:/Users/Qi/Dropbox/Toulouse2017/npy/A/VV/mean_map.npy')
mean_map_VH=np.load('C:/Users/Qi/Dropbox/Toulouse2017/npy/A/VH/mean_map.npy')
variance_map_VV=np.load('C:/Users/Qi/Dropbox/Toulouse2017/npy/A/VV/variance_map.npy')
variance_map_VH=np.load('C:/Users/Qi/Dropbox/Toulouse2017/npy/A/VH/variance_map.npy')
l_map_VV=np.load('C:/Users/Qi/Dropbox/Toulouse2017/npy/A/VV/l_map.npy')
l_map_VH=np.load('C:/Users/Qi/Dropbox/Toulouse2017/npy/A/VH/l_map.npy')


indicator_fields=np.empty([2001,2001])
for x in xrange(2001):
    for y in xrange(2001):
        if mask[x,y]==1:
            indicator_fields[x,y]=np.nan
        else:
            if mean_map_VV[x,y]>-8.5 or mean_map_VH[x,y]>-15:#forest and urban
                indicator_fields[x,y]=np.nan
            else:
                if mean_map_VV[x,y]<0.171*variance_map_VV[x,y]-13.757 or mean_map_VH[x,y]<-0.133*variance_map_VH[x,y]-19.545:#nonirrigated
                    indicator_fields[x,y]=4
                else:
                    if variance_map_VV[x,y]<0.03*l_map_VV[x,y]+0.82 and variance_map_VH[x,y]<0.008*l_map_VH[x,y]+1.361:#irrigated trees 
                        indicator_fields[x,y]=2
                    else:
                        #if variance_map_VV[x,y]>0.066*l_map_VV[x,y]+2.476 or variance_map_VH[x,y]>0.271*l_map_VH[x,y]-2.239:
                        indicator_fields[x,y]=1 #irrigated crops

#==============================================================================
# indicator_fields[np.isnan(indicator_fields)]=0
#==============================================================================
plt.figure()
plt.imshow(indicator_fields,cmap='jet',vmin=1, vmax=4) 
#==============================================================================
# plt.colorbar()
#==============================================================================

empty_file = 'C:/Users/Qi/Dropbox/Toulouse2017/fields/empty.tif'
out_dir = "C:/Users/Qi/Dropbox/Toulouse2017/"
dataset = gdal.Open(empty_file)
geotransform = dataset.GetGeoTransform()
backscatter_empty = dataset.GetRasterBand(1).ReadAsArray()
projection = dataset.GetProjection()
dataset = None

out_file = os.path.join(out_dir, "classification.tif")
driver = gdal.GetDriverByName('GTiff')
dataset = driver.Create(out_file, backscatter_empty.shape[1], backscatter_empty.shape[0], 1, gdal.GDT_Float32)
dataset.SetGeoTransform(geotransform)
dataset.SetProjection(projection)
dataset.GetRasterBand(1).WriteArray(indicator_fields, 0, 0)
dataset = None 

class_mask=("C:/Users/Qi/Dropbox/Toulouse2017/fields/urgell_class.tif")
dataset = gdal.Open(class_mask)
geotransform = dataset.GetGeoTransform()
classes = dataset.GetRasterBand(1).ReadAsArray()
projection = dataset.GetProjection()
dataset = None
classes=classes.astype(float)

classes[classes==3.0]=np.nan
classes[classes==4.0]=np.nan

plt.figure()
plt.imshow(classes,cmap='jet',vmin=1, vmax=2.5) 
plt.colorbar()

#==============================================================================
# indicator_fields[indicator_fields==2]=1
# indicator_fields[indicator_fields==4]=2
# indicator_fields[indicator_fields==0]=np.nan
# tr=0
# total_n=0
# for x in xrange(2001):
#     for y in xrange(2001):
#         if not np.isnan(indicator_fields[x,y]) and not np.isnan(classes[x,y]):
#             total_n=total_n+1
#         if indicator_fields[x,y]==classes[x,y]:
#             tr=tr+1
# 
# acc=100.0*tr/total_n
# print(acc)
#==============================================================================

