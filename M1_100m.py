# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 11:45:10 2022

@author: Qi
"""

import glob
import numpy as np
import gdal
import os
from datetime import datetime
import scipy.interpolate

def db2linear(data):
    #data = filters.median_filter(data, size=7)
    data = 10**(data/10)
    return data
def linear2db(data):
    #data = filters.median_filter(data, size=7)
    data = 10*np.log10(data)
    return data
    
p_size=str('100m')
pol=str('VV')
def clear_all():
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]


track=str('133')
hh=19
mm=18

bc_files = glob.glob("D:/01_Projects/01_EO_Africa/Data/S1/VV/*.tif")
bc_files.sort()
ndvi = np.load('D:/01_Projects/01_EO_Africa/Data/npy/s2_ndvi_interp_daily.npy')
date_ndvi=np.load('D:/01_Projects/01_EO_Africa/Data/npy/date_list_daily.npy')

n_bc = len(bc_files)
s2_ndvi=np.zeros((200,200,n_bc))*np.nan

 
for i in range(n_bc):
    bc_file = bc_files[i]
    date_str = os.path.basename(bc_file).split("_")[4].split('T')[0]
    ymd=list(date_str)
    y=("%s%s%s%s"%(ymd[0],ymd[1],ymd[2],ymd[3]))
    m=("%s%s"%(ymd[4],ymd[5]))
    d=("%s%s"%(ymd[6],ymd[7]))
    y=int(y)
    m=int(m)
    d=int(d)
    date_bc=datetime(y,m,d)
    idx=np.where(date_ndvi==datetime.toordinal(date_bc))[0][0]
    s2_ndvi[:,:,i]= ndvi[:,:,idx]
  
np.save('D:/01_Projects/01_EO_Africa/Data/npy/s2_ndvi_%s.npy'%track,s2_ndvi)    


in_files = glob.glob("D:/01_Projects/01_EO_Africa/Data/S1/VV/*.tif")
ndvi = np.load('D:/01_Projects/01_EO_Africa/Data/npy/s2_ndvi_%s.npy'%track)
in_files.sort()
n = len(in_files)
ndvi_class=np.zeros((20,1))*np.nan
for i in range(20):
    ndvi_class[i]=[0.1*i/2]
 
bc_min=np.zeros((200,200,19))

for p in range(19):
    for i in range(n):
        bc_file= in_files[i]
        date_str = os.path.basename(bc_file).split("_")[4].split('T')[0]
        date_bc=int(date_str)
        
        dataset = gdal.Open(bc_file)
        geotransform = dataset.GetGeoTransform()
        bc = dataset.GetRasterBand(1).ReadAsArray()
        dataset = None
        bc = db2linear(bc)
        
        av_bc=np.zeros((200,200))*np.nan
        
        for x in range(200):
            for y in range(200):
                xl=x*10
                xr=(x+1)*10
                yb=y*10
                yu=(y+1)*10
                
                bc_win=bc[xl:xr,yb:yu]
                av_bc[x,y]=np.mean(bc_win)
                if not np.isnan(av_bc[x,y]):
                    av_bc[x,y]=linear2db(av_bc[x,y])
            
                if not np.isnan(ndvi[x,y,i]) and ndvi[x,y,i]>ndvi_class[p] and ndvi[x,y,i]<ndvi_class[p+1]:
                    if av_bc[x,y] < bc_min[x,y,p]:
                        bc_min[x,y,p] = av_bc[x,y] 


np.save('D:/01_Projects/01_EO_Africa/Data/npy/bc_min_s2_ndvi_19classes',bc_min)
print('bc_min saved')

#%%
bc_min = np.load('D:/01_Projects/01_EO_Africa/Data/npy/bc_min_s2_ndvi_19classes.npy')
n = len(in_files)
diff_bc=np.zeros((200,200,n))*np.nan
bc_min[bc_min==0] = np.nan
for p in range(19):
    for i in range(n):
        # read the first bc file
        bc_file = in_files[i]
        dataset = gdal.Open(bc_file)
        geotransform = dataset.GetGeoTransform()
        bc = dataset.GetRasterBand(1).ReadAsArray()
        dataset = None
        
        
        for x in range(200):
            for y in range(200):
                xl=x*10
                xr=(x+1)*10
                yb=y*10
                yu=(y+1)*10
                
                bc_win=bc[xl:xr,yb:yu]
                bc_win = db2linear(bc_win)
                av_bc=np.mean(bc_win)
                if not np.isnan(av_bc):
                    av_bc=linear2db(av_bc)
            
                if ndvi[x,y,i]>ndvi_class[p] and ndvi[x,y,i]<ndvi_class[p+1]:
                    
                    av_min=bc_min[x,y,p]
                    if ~np.isnan(av_bc) and av_min<0:
                        diff_bc[x,y,i] = av_bc - av_min    
    
np.save('D:/01_Projects/01_EO_Africa/Data/npy/diff_bc_s2_ndvi_19classes',diff_bc)
print('diff_bc saved')
    
#%%    
#==============================================================================
# log plot
#==============================================================================
import numpy as np
from matplotlib import pyplot as plt
from heapq import nlargest
import math
from scipy.optimize import leastsq

track=str('133')
p_size=str('100m')
bc = np.load('D:/01_Projects/01_EO_Africa/Data/npy/diff_bc_s2_ndvi_19classes.npy')
ndvi = np.load('D:/01_Projects/01_EO_Africa/Data/npy/s2_ndvi_%s.npy'%track)
n = len(bc[1,1,:])


#==============================================================================
#     ndvi[ndvi > 0.5] = np.nan
#==============================================================================
# ndvi_max=0.8#0.5
# ndvi_min=0
# ndvi_step=0
# for p in range(20):
#     ndvi_step=np.append(ndvi_step,ndvi_min+(ndvi_max-ndvi_min)/20*(p+1))  


# pct_del=0.001
# n = len(bc[1,1,:])
# bc_step=0.2

# for p in range(20):
#     ndvi_lst=[]
#     bc_lst=[]
#     index_i=[]
#     index_j=[]
#     index_k=[]
#     for i in range(200):
#         for j in range(200):
#             for k in range(n):
#                 if not np.isnan(bc[i,j,k]) and not np.isnan(ndvi[i,j,k]):
#                     if ndvi[i,j,k] != 0:
#                         if ndvi[i,j,k]>=ndvi_step[p] and ndvi[i,j,k]<=ndvi_step[p+1]:
#                             ndvi_lst=np.append(ndvi_lst, ndvi[i,j,k])
#                             bc_lst=np.append(bc_lst, bc[i,j,k])   
#                             index_i=np.append(index_i,i)
#                             index_j=np.append(index_j,j)
#                             index_k=np.append(index_k,k)
#     if bc_lst!=[]:
        
#         n_step=len(bc_lst)
#         dens=0
#         bc_lst.sort
#         bc_max=np.max(bc_lst)
#         bc_min=0
#         bc_class=0
#         for k in range(20):
#             bc_class=np.append(bc_class,bc_min+(bc_max-bc_min)/20*(k+1))
        
#         #bc_class=[bc_max,bc_max-1,bc_max-2,bc_max-3,bc_max-4,bc_max-5,bc_max-6,bc_max-7,bc_max-8,bc_max-9,bc_max-10]
         
#         for i_bc in range(20):
#             index_ii=[]
#             index_jj=[]
#             index_kk=[]
#             for k in range(n_step):
#                 if bc_lst[k]>=bc_class[20-i_bc-1] and bc_lst[k]<=bc_class[20-i_bc]:
#                     dens=dens+1
#                     index_ii=np.append(index_ii,index_i[k])
#                     index_jj=np.append(index_jj,index_j[k])
#                     index_kk=np.append(index_kk,index_k[k])
#             if dens<20:
#                 for l in range(len(index_ii)):
#                     bc[int(index_ii[l]),int(index_jj[l]),int(index_kk[l])] = np.nan


plt.figure()
plt.title("Backscatter difference w.r.t. NDVI at 100m resolution",fontsize=20)
plt.xlabel("NDVI",fontsize=20)
plt.ylabel("Backscatter difference",fontsize=20)
plt.grid(True)
       
for i in range(200):
    for j in range(200):
        plt.plot(ndvi[i,j,:],bc[i,j,:],'ro')
#==============================================================================
# 
#%%
#==============================================================================
ndvi_max=0.8#0.85
ndvi_min=0.05
ndvi_step=0.05
for p in range(20):
    ndvi_step=np.append(ndvi_step,ndvi_min+(ndvi_max-ndvi_min)/20*(p+1))  
x=[]
y=[]
ndvi_mean=[]
bc_mean=[]  
ndvi_top=[]
bc_top=[]
pct_max=0.01
for p in range(20):
    ndvi_lst=[]
    bc_lst=[]
    for i in range(200):
        for j in range(200):
            for k in range(n):
                if not np.isnan(bc[i,j,k]) and not np.isnan(ndvi[i,j,k]):
                    if ndvi[i,j,k] != 0:
                        if ndvi[i,j,k]>ndvi_step[p] and ndvi[i,j,k]<ndvi_step[p+1]:
                            ndvi_lst=np.append(ndvi_lst, ndvi[i,j,k])
                            bc_lst=np.append(bc_lst, bc[i,j,k]) 
# =============================================================================
#     num=int(math.ceil(len(bc_lst)*pct_max))
# =============================================================================
    num=20   
    data=nlargest(num, enumerate(bc_lst),key=lambda x:x[1])
    indices, vals = zip(*data)
    
    for ii in range(len(indices)):
        ndvi_top=np.append(ndvi_top,ndvi_lst[indices[ii]])
        bc_top=np.append(bc_top,bc_lst[indices[ii]])


del x,y
x=ndvi_top
y=bc_top
plt.plot(x,y,'go')


funcQuad=lambda tpl,x : tpl[0]*x+tpl[1]
func=funcQuad
ErrorFunc=lambda tpl,x,y: func(tpl,x)-y
tplInitial=(1.0,2.0)
tplFinal,success=leastsq(ErrorFunc,tplInitial[:],args=(x,y))
xx=np.linspace(x.min(),x.max(),50)
yy=func(tplFinal,xx)
plt.plot(xx,yy,'b-')
plt.ylim([0,20])
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20) 
plt.text(0.35, 19,"y = %s x + %s"%(round(tplFinal[0],2),round(tplFinal[1],2)),
     horizontalalignment='left',
     verticalalignment='top', fontsize=15)
print ("track %s: y = %s x + %s"%(track,tplFinal[0],tplFinal[1]))