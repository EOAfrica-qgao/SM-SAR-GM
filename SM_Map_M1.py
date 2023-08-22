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

bc_min = np.load('D:/01_Projects/01_EO_Africa/Data/npy/bc_min_s2_ndvi_19classes.npy')
ndvi = np.load('D:/01_Projects/01_EO_Africa/Data/npy/s2_ndvi_%s.npy'%track)
def f_ndvi(ndvi):
    data = -23.197606021892618 *ndvi + 17.63255650873604
        #data = -12.543811224712572 x + 21.64099929653802 #ndvi<0.6
    return data

    
Mv_min=0.0
Mv_max=0.3
bc_min[bc_min==0] = np.nan
def db2linear(data):
    #data = filters.median_filter(data, size=7)
    data = 10**(data/10)
    return data
def linear2db(data):
    #data = filters.median_filter(data, size=7)
    data = 10*np.log10(data)
    return data

bc_files = glob.glob("D:/01_Projects/01_EO_Africa/Data/S1/VV/*.tif")
bc_files.sort()
n= len(bc_files)


ndvi_class=np.zeros((20,1))*np.nan
for i in range(20):
    ndvi_class[i]=[0.1*i/2]
    
Mv=np.zeros((200,200,n))*np.nan
date_bc=np.empty((n,0)).tolist()
av_bc=np.zeros((200,200,n))*np.nan
for i in range(n):
    
    bc_file= bc_files[i]
    date_str = os.path.basename(bc_file).split("_")[4].split('T')[0]
    ymd=list(date_str)
    y=("%s%s%s%s"%(ymd[0],ymd[1],ymd[2],ymd[3]))
    m=("%s%s"%(ymd[4],ymd[5]))
    d=("%s%s"%(ymd[6],ymd[7]))
    y=int(y)
    m=int(m)
    d=int(d)
    date_bc[i]=datetime(y,m,d)
    
    dataset = gdal.Open(bc_file)
    geotransform = dataset.GetGeoTransform()
    bc = dataset.GetRasterBand(1).ReadAsArray()#in dB
    dataset = None
    bc = db2linear(bc)
    
    
    
    for x in range(200):
        for y in range(200):
            xl=x*10
            xr=(x+1)*10
            yb=y*10
            yu=(y+1)*10
            
            bc_win=bc[xl:xr,yb:yu]
            
            av_bc[x,y,i]=np.mean(bc_win)
            if not np.isnan(av_bc[x,y,i]):
                av_bc[x,y,i]=linear2db(av_bc[x,y,i])
    
            f=f_ndvi(ndvi[x,y,i])
            bc_minimum=np.nan
            for p in range(9):
                if ~np.isnan(ndvi[x,y,i]): #and srtm[x,y-200]<500 
                    if ndvi[x,y,i]>ndvi_class[p] and ndvi[x,y,i]<ndvi_class[p+1]:# and bc_min[x,y-200,p]>-15:
                        bc_minimum=bc_min[x,y,p]
            if ~np.isnan(bc_minimum):
                bc_diff=av_bc[x,y,i]-bc_minimum
                if bc_diff>f:
                    bc_diff=f
                Mv[x,y,i]=(av_bc[x,y,i]-bc_minimum)/f*Mv_max+Mv_min
                if Mv[x,y,i]<Mv_min:
                    Mv[x,y,i]=Mv_min
                if Mv[x,y,i]>Mv_max:
                    Mv[x,y,i]=Mv_max

np.save('D:/01_Projects/01_EO_Africa/Data/npy/Mv_map_retrieved_m1.npy',Mv)

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

Mv=np.load('D:/01_Projects/01_EO_Africa/Data/npy/Mv_map_retrieved_m1.npy')
for i in range(78,120):
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
    
    plt.savefig("D:/01_Projects/01_EO_Africa/Data/figure/SM_M1/%s.png"%(datetime.date(date_bc[i])),dpi=800)
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

Mv=np.load('D:/01_Projects/01_EO_Africa/Data/npy/Mv_map_retrieved_m1.npy')
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
    with rasterio.open('D:/01_Projects/01_EO_Africa/Data/tiff/%s.tif'%(datetime.date(date_bc[i])), 'w', **metadata) as dst:
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
