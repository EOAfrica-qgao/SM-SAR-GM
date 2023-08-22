# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 14:40:19 2016

@author: gaoq
"""


import glob
import numpy as np
import gdal
import os
from datetime import datetime
from matplotlib import pyplot as plt
import pandas as pd

track=str('133')
p_size=str('100m')
def db2linear(data):
    #data = filters.median_filter(data, size=7)
    data = 10**(data/10)
    return data
def linear2db(data):
    #data = filters.median_filter(data, size=7)
    data = 10*np.log10(data)
    return data


#==============================================================================
#              initialize   
#==============================================================================

def f_ndvi_pos(ndvi):
    
    #data = -10*ndvi+15.36
    data = -14.803574070420243 *ndvi + 11.655839523070727
    return data
def f_ndvi_neg(ndvi):
    #data = 10*ndvi-15.36
    data = 14.803574070420243 *ndvi - 11.655839523070727
    return data

Mv_diff_max=0.12

bc_files = glob.glob("D:/01_Projects/01_EO_Africa/Data/S1/VV/*.tif")
bc_files.sort()
ndvi = np.load('D:/01_Projects/01_EO_Africa/Data/npy/s2_ndvi_133.npy')

n= len(bc_files)


Mv=np.zeros((200,200,n))*np.nan
Mv_retrieved=np.load('D:/01_Projects/01_EO_Africa/Data/npy/Mv_map_retrieved_m1.npy')
date_bc=np.empty((n,0)).tolist()
n_max=78
date_bc[n_max]=datetime(2018,8,18)
Mv[:,:,n_max]=Mv_retrieved[:,:,n_max]

#bc_min=np.zeros((200,200,3))
diff_bc=np.zeros((200,200,n))*np.nan
ndvi_mean=np.zeros((200,200,n))*np.nan
for i in range(n_max+1,n):
    bc_file1= bc_files[i-1]
    date_str = os.path.basename(bc_file1).split("_")[4].split('T')[0]
    ymd=list(date_str)
    y=("%s%s%s%s"%(ymd[0],ymd[1],ymd[2],ymd[3]))
    m=("%s%s"%(ymd[4],ymd[5]))
    d=("%s%s"%(ymd[6],ymd[7]))
    y=int(y)
    m=int(m)
    d=int(d)
    date_bc1=datetime(y,m,d)
    
    
    dataset = gdal.Open(bc_file1)
    geotransform = dataset.GetGeoTransform()
    bc1 = dataset.GetRasterBand(1).ReadAsArray()
    dataset = None
    bc1 = db2linear(bc1)
    
    bc_file2= bc_files[i]
    date_str = os.path.basename(bc_file2).split("_")[4].split('T')[0]
    ymd=list(date_str)
    y=("%s%s%s%s"%(ymd[0],ymd[1],ymd[2],ymd[3]))
    m=("%s%s"%(ymd[4],ymd[5]))
    d=("%s%s"%(ymd[6],ymd[7]))
    y=int(y)
    m=int(m)
    d=int(d)
    date_bc2=datetime(y,m,d)
    
    
    dataset = gdal.Open(bc_file2)
    geotransform = dataset.GetGeoTransform()
    bc2 = dataset.GetRasterBand(1).ReadAsArray()
    dataset = None
    bc2 = db2linear(bc2)  
    
    av_bc1=np.zeros((200,200))*np.nan
    av_bc2=np.zeros((200,200))*np.nan
    
    
    
    for x in range(200):
        for y in range(200):
            xl=x*10
            xr=(x+1)*10
            yb=y*10
            yu=(y+1)*10
            
            bc_win1=bc1[xl:xr,yb:yu]
            av_bc1[x,y]=np.mean(bc_win1)
            bc_win2=bc2[xl:xr,yb:yu]
            av_bc2[x,y]=np.mean(bc_win2)
            if not np.isnan(av_bc1[x,y]) and not np.isnan(av_bc2[x,y]) and not np.isnan(Mv[x,y,i-1]):# and srtm[x,y]<500:
                if 3<datetime.toordinal(date_bc2)-datetime.toordinal(date_bc1)<13:
                    diff_bc[x,y,i]=linear2db(av_bc2[x,y])-linear2db(av_bc1[x,y])
                    ndvi_mean[x,y,i]=(ndvi[x,y,i-1]+ndvi[x,y,i])/2
                    if diff_bc[x,y,i]>0:
                        f=f_ndvi_pos(ndvi_mean[x,y,i])
                        Mv[x,y,i]=diff_bc[x,y,i]/f*Mv_diff_max+Mv[x,y,i-1]
                        if Mv[x,y,i]<0:
                            Mv[x,y,i]=0
                        if Mv[x,y,i]>0.3:
                            Mv[x,y,i]=0.3
                    if diff_bc[x,y,i]<0:
                        f=f_ndvi_neg(ndvi_mean[x,y,i])
                        Mv[x,y,i]=diff_bc[x,y,i]/f*(-Mv_diff_max)+Mv[x,y,i-1]
                        if Mv[x,y,i]<0:
                            Mv[x,y,i]=0
                        if Mv[x,y,i]>0.3:
                            Mv[x,y,i]=0.3
    date_bc[i]=date_bc2

np.save('D:/01_Projects/01_EO_Africa/Data/npy/Mv_map_retrieved_m2.npy',Mv)


#%%
from pykml import parser
from mpl_toolkits.axes_grid1 import make_axes_locatable
mask_kml = ("D:/01_Projects/01_EO_Africa/AOI.kml")
a=parser.parse(mask_kml).getroot().Document.Folder.Placemark.MultiGeometry.Polygon.outerBoundaryIs.LinearRing.coordinates
a=str(a)
aa=a.split('\t')[0].split(" ")
n_points=len(aa)-1
mask_coor=np.zeros((n_points,2))*np.nan #lon lat
for i in range(n_points):
    mask_coor[i,0]=float(aa[i].split(",")[0])
    mask_coor[i,1]=float(aa[i].split(",")[1])
extent = (mask_coor[0,0], mask_coor[1,0],
          mask_coor[0,1], mask_coor[3,1])

def major_formatter(x, pos):
    return "%.2f" % x
# Mv=np.load('D:/01_Projects/01_EO_Africa/Data/npy/Mv_map_retrieved_m1.npy')
# for i in range(120,151):
#     fig = plt.figure()  
#     cmap = plt.cm.jet
#     SM1=Mv[:,:,i]
#     im = plt.imshow(SM1,cmap='jet_r', vmin=0, vmax=0.25) 
#     cbar = plt.colorbar(im)
#     cbar.ax.tick_params(labelsize=15)
#     plt.title("Soil Moisture Retrieved (%s)"%(datetime.date(date_bc[i])),fontsize=20) 

Mv=np.load('D:/01_Projects/01_EO_Africa/Data/npy/Mv_map_retrieved_m2.npy')
for i in range(78,151):
    plt.figure(figsize=(15,15))
    plt.subplot(1, 1, 1)
    cmap = plt.cm.jet
    SM1=Mv[:,:,i]
    img = plt.imshow(SM1, extent=extent,origin='upper',cmap='jet_r', vmin=0, vmax=0.25)
    ax = plt.gca()
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    
    im_ratio = SM1.shape[0]/SM1.shape[1]
    cbar = plt.colorbar(img,fraction=0.0448*im_ratio, pad=0.04)
    cbar.ax.tick_params(labelsize=20)
    plt.xticks(fontsize = 20) 
    plt.yticks(fontsize = 20)
    ax.xaxis.set_major_formatter(major_formatter)
    ax.yaxis.set_major_formatter(major_formatter)
    plt.title("Soil Moisture Retrieved (%s)"%(datetime.date(date_bc[i])),fontsize=25)
    #==============================================================================
    # plt.rc('xtick', labelsize=16)      
    # plt.rc('ytick', labelsize=16) 
    #==============================================================================
    
    # plt.axis('tight')
    plt.xlabel("Longitude [deg]",fontsize=20)
    plt.ylabel("Latitude [deg]",fontsize=20)
    
    plt.savefig("D:/01_Projects/01_EO_Africa/Data/figure/SM_M2/%s.png"%(datetime.date(date_bc[i])),dpi=800)
#==============================================================================
# fig = plt.figure()   
# plt.hold(True)
# bc1=av_bc[:,:,i]
# plt.imshow(bc1,cmap='jet', vmin=-14, vmax=-4) 
# #plt.gca().invert_yaxis()
# plt.colorbar() 
# plt.title("Backscatter of Sentinel 1 (%s)"%(date_bc[i]),fontsize=20)
# 
# fig = plt.figure()   
# plt.hold(True)
# i=29
# cmap = plt.cm.jet
# NDVI=ndvi[:,:,i]
# plt.imshow(NDVI,cmap='jet_r',vmin=0.1, vmax=0.8) 
# plt.colorbar() 
# plt.title("NDVI (%s)"%(date_bc[i]),fontsize=20) 
#==============================================================================


#%%
import rasterio

Mv=np.load('D:/01_Projects/01_EO_Africa/Data/npy/Mv_map_retrieved_m2.npy')
for i in range(78,151):
    SM1=Mv[:,:,i]
    
    data=SM1
    
    # Define the transform with the specified origin, pixel size, and size
    transform = rasterio.Affine(100, 0, 335000, 0, -100, 1613300)
    
    # Define the metadata for the output file
    metadata = {'driver': 'GTiff',
                'height': data.shape[0],
                'width': data.shape[1],
                'count': 1,
                'dtype': data.dtype,
                'crs': 'EPSG:32628',
                'transform': transform}
    
    # Write the data to a georeferenced TIFF file with the specified metadata
    with rasterio.open('D:/01_Projects/01_EO_Africa/Data/tiff/M2/%s.tif'%(datetime.date(date_bc[i])), 'w', **metadata) as dst:
        dst.write(data, 1)
    
#%%
# Mv=np.load('E:/isardSAT_work/0_SM_QI/code 100m/npy/Mv_retrieved_m2_110_Mvmax0.35_diffmax_0.2.npy')
# i=29
# SM1=Mv[:,:,i]

# data=SM1

# # Define the transform with the specified origin, pixel size, and size
# transform = rasterio.Affine(100, 0, 326410, 0, -100, 4641300)

# # Define the metadata for the output file
# metadata = {'driver': 'GTiff',
#             'height': data.shape[0],
#             'width': data.shape[1],
#             'count': 1,
#             'dtype': data.dtype,
#             'crs': 'EPSG:32631',
#             'transform': transform}

# # Write the data to a georeferenced TIFF file with the specified metadata
# with rasterio.open('D:/01_Projects/20150902.tif', 'w', **metadata) as dst:
#     dst.write(data, 1)
