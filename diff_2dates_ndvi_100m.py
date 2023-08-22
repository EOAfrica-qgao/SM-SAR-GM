# -*- coding: utf-8 -*-
"""
Created on Wed May 03 17:36:49 2017

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
from heapq import nlargest
import math
from scipy.optimize import leastsq

p_size=str('100m')
Mv_diff_max=0.3

track=str('1')
hh=6
mm=0


def db2linear(data):
    #data = filters.median_filter(data, size=7)
    data = 10**(data/10)
    return data
def linear2db(data):
    #data = filters.median_filter(data, size=7)
    data = 10*np.log10(data)
    return data


# =============================================================================
# bc_files = glob.glob("D:/01_Projects/01_EO_Africa/Data/S1/VV/*.tif")
# bc_files.sort()
# ndvi = np.load('D:/01_Projects/01_EO_Africa/Data/npy/s2_ndvi_133.npy')
# # ndvi = np.load('D:/01_Projects/01_EO_Africa/Data/npy/s2_ndvi_interp_daily.npy')
# 
# n = len(bc_files)
# 
# diff_bc=np.zeros((200,200,n-1))*np.nan
# ndvi_mean=np.zeros((200,200,n-1))*np.nan
# date_bc=[]#np.empty((n,0)).tolist()
# for i in range(n-1):
#     bc_file1 = bc_files[i]
#     date_str = os.path.basename(bc_file1).split("_")[4].split('T')[0]
#     ymd=list(date_str)
#     y=("%s%s%s%s"%(ymd[0],ymd[1],ymd[2],ymd[3]))
#     m=("%s%s"%(ymd[4],ymd[5]))
#     d=("%s%s"%(ymd[6],ymd[7]))
#     y=int(y)
#     m=int(m)
#     d=int(d)
#     date_bc1=datetime(y,m,d)
#     
#     dataset = gdal.Open(bc_file1)
#     geotransform = dataset.GetGeoTransform()
#     bc1 = dataset.GetRasterBand(1).ReadAsArray()
#     dataset = None
#     bc1 = db2linear(bc1)
#     
#     bc_file2= bc_files[i+1]
#     date_str = os.path.basename(bc_file2).split("_")[4].split('T')[0]
#     ymd=list(date_str)
#     y=("%s%s%s%s"%(ymd[0],ymd[1],ymd[2],ymd[3]))
#     m=("%s%s"%(ymd[4],ymd[5]))
#     d=("%s%s"%(ymd[6],ymd[7]))
#     y=int(y)
#     m=int(m)
#     d=int(d)
#     
#     date_bc2=datetime(y,m,d)
#     
#     
#     dataset = gdal.Open(bc_file2)
#     geotransform = dataset.GetGeoTransform()
#     bc2 = dataset.GetRasterBand(1).ReadAsArray()
#     dataset = None
#     bc2 = db2linear(bc2)  
#     
#     av_bc1=np.zeros((200,200))*np.nan
#     av_bc2=np.zeros((200,200))*np.nan
#     
#     
#     date_bc=np.append(date_bc,date_bc2)
#     for x in range(200):
#         for y in range(200):
#             #if ~np.isnan(field_mask[x,y]):
#             xl=x*10
#             xr=(x+1)*10
#             yb=y*10
#             yu=(y+1)*10
#             
#             bc_win1=bc1[xl:xr,yb:yu]
#             av_bc1[x,y]=np.mean(bc_win1)
#             bc_win2=bc2[xl:xr,yb:yu]
#             av_bc2[x,y]=np.mean(bc_win2)
#             if not np.isnan(av_bc1[x,y]) and not np.isnan(av_bc2[x,y]):
#                 if 1<datetime.toordinal(date_bc2)-datetime.toordinal(date_bc1)<13: # and abs(ndvi[x,y,i]-ndvi[x,y,i+1])<0.1:
#                     diff_bc[x,y,i]=linear2db(av_bc2[x,y])-linear2db(av_bc1[x,y])
#                     ndvi_mean[x,y,i]=(ndvi[x,y,i]+ndvi[x,y,i+1])/2
# 
# np.save('D:/01_Projects/01_EO_Africa/Data/npy/Diff_bc_dates_m2.npy',diff_bc)
# np.save('D:/01_Projects/01_EO_Africa/Data/npy/ndvi_mean_s2_m2.npy',ndvi_mean)
# np.save('D:/01_Projects/01_EO_Africa/Data/npy/date_bc_m2.npy',date_bc)
# print('bc_diff saved')
# =============================================================================
#%%
import glob
import numpy as np
import gdal
import numpy as np
import os
from datetime import datetime
from matplotlib import pyplot as plt

pol=str('VV')
diff_bc=np.load('D:/01_Projects/01_EO_Africa/Data/npy/Diff_bc_dates_m2.npy', allow_pickle=True)
ndvi_mean=np.load('D:/01_Projects/01_EO_Africa/Data/npy/ndvi_mean_s2_m2.npy', allow_pickle=True)
date_bc=np.load('D:/01_Projects/01_EO_Africa/Data/npy/date_bc_m2.npy', allow_pickle=True)
# =============================================================================
# plt.figure()
# n = len(diff_bc[1,1,:])      
# for k in range(n-1):
#     if date_bc[k]<datetime(2018,4,1) or datetime(2018,5,31)<date_bc[k]:
#         for i in range(200):
#             for j in range(200):
#                 red_points, =plt.plot(ndvi_mean[i,j,k],diff_bc[i,j,k],'ro',label='red')
#     print (k)
# plt.title("Backscatter difference between 2 adjacent days w.r.t. NDVI (track %s)"%(track),fontsize=20)
# plt.xlabel("NDVI",fontsize=20)
# plt.ylabel("Backscatter difference",fontsize=20)
# plt.legend([red_points], ['exclude April May'])
# =============================================================================
  


# =============================================================================
# diff_bc=np.load('C:/Users/Qi/Desktop/SM_QI/code 100m/npy/diff_bc_dates_%s_20km.npy'%(track))
# ndvi_mean=np.load('C:/Users/Qi/Desktop/SM_QI/code 100m/npy/ndvi_mean_%s_s2_20km.npy'%(track))
# date_bc=np.load('C:/Users/Qi/Desktop/SM_QI/code 100m/npy/date_bc_m2_%s_20km.npy'%track)
# srtm_file='C:/Users/Qi/Desktop/SM_QI/tif/SRTM.tif'
# driver = gdal.GetDriverByName('GTiff')
# dataset = gdal.Open(srtm_file)
# geotransform = dataset.GetGeoTransform()
# srtm = dataset.GetRasterBand(1).ReadAsArray()
# n=len(diff_bc[1,1,:])
# =============================================================================
#==============================================================================
#     for i in range(200):
#         for j in range(200):
#             for k in range(n):
#                 if srtm[i,j]>500:
#                     diff_bc[i,j,k]=np.nan
#==============================================================================
#==============================================================================
#     for k in xrange(n-1):
#         if date_bc[k]>datetime(2016,11,1):
# #==============================================================================
# #         if datetime(2015,3,31)<date_bc[k]<datetime(2015,6,1) or datetime(2016,3,31)<date_bc[k]<datetime(2016,6,1):
# #==============================================================================
#             for i in range(200):
#                 for j in range(200):
#                     diff_bc[i,j,k]=np.nan
#==============================================================================



# ndvi_lst=[]
# for i in range(200):
#     for j in range(200):
#         for k in range(n):
#             if not np.isnan(diff_bc[i,j,k]) and not np.isnan(ndvi_mean[i,j,k]):
#                 if ndvi_mean[i,j,k] != 0:
#                     ndvi_lst=np.append(ndvi_lst, ndvi_mean[i,j,k])                     
# ndvi_max=np.max(ndvi_lst)
# ndvi_min=np.min(ndvi_lst)

ndvi_max=0.8
ndvi_min=0.07
ndvi_step=ndvi_min
ndvi_class=ndvi_min
for p in range(20):
    ndvi_step=np.append(ndvi_step,ndvi_min+(ndvi_max-ndvi_min)/20*(p+1))  



#==============================================================================
#     pct_del=0.001
#     #pct_max=0.25
#     n = len(diff_bc[1,1,:])
#     bc_step=0.2
#     for p in range(20):
#         ndvi_lst=[]
#         bc_lst=[]
#         bc_lst_abs=[]
#         for i in range(200):
#             for j in range(200):
#                 for k in range(n):
#                     if not np.isnan(diff_bc[i,j,k]):
#                         if ndvi_mean[i,j,k] != 0:
#                             if ndvi_mean[i,j,k]>=ndvi_step[p] and ndvi_mean[i,j,k]<=ndvi_step[p+1]:
#                                 ndvi_lst=np.append(ndvi_lst, ndvi_mean[i,j,k])
#                                 bc_lst=np.append(bc_lst, diff_bc[i,j,k]) 
#                                 bc_lst_abs=np.append(bc_lst_abs, abs(diff_bc[i,j,k]))
#         bc_max=np.max(bc_lst_abs)
#         n_step=len(ndvi_lst)
#         dens=0
#       
#         for i_bc in range(20):
#             bc_del=[]
#             indices_del=[]
#             for j in range(n_step):
#                 if bc_lst_abs[j]>=bc_max-bc_step*(i_bc+1) and bc_lst_abs[j]<=bc_max-bc_step*i_bc:
#                     dens=dens+1
#                     bc_del=np.append(bc_del, bc_lst[j]) 
#                     indices_del=np.append(indices_del, int(j))
#             if dens<15:
#                 
#                 for d in range(len(bc_del)):
#                     #plt.plot(ndvi_lst[indices_del[d]],bc_lst[indices_del[d]],'ko')
#                     for i in range(200):
#                         for j in range(200):
#                             for k in range(n):
#                                 if diff_bc[i,j,k]==bc_del[d]:
#                                     diff_bc[i,j,k]=np.nan
#     
#         del bc_max
#         del bc_del
#==============================================================================
plt.figure()  
for i in range(200):
    for j in range(200):
        plt.plot(ndvi_mean[i,j,:],diff_bc[i,j,:],'ro')


#%%==============================================================================

x=[]
y=[] 
ndvi_top_pos=[]
ndvi_top_neg=[]
bc_top_pos=[]
bc_top_neg=[]
pct_max=0.01
n = len(diff_bc[1,1,:]) 
for p in range(20):
    ndvi_lst_pos=[]
    bc_lst_pos=[]
    ndvi_lst_neg=[]
    bc_lst_neg=[]
    for i in range(200):
        for j in range(200):
            for k in range(n):
                if not np.isnan(diff_bc[i, j, k]) and ndvi_mean[i, j, k] > 0:
                    if ndvi_step[p] < ndvi_mean[i, j, k] < ndvi_step[p+1]:
                        if diff_bc[i, j, k] > 0:
                            ndvi_lst_pos.append(ndvi_mean[i, j, k])
                            bc_lst_pos.append(diff_bc[i, j, k])
                        elif diff_bc[i, j, k] < 0:
                            ndvi_lst_neg.append(ndvi_mean[i, j, k])
                            bc_lst_neg.append(diff_bc[i, j, k])
    ndvi_lst_pos = np.array(ndvi_lst_pos)
    bc_lst_pos = np.array(bc_lst_pos)
    ndvi_lst_neg = np.array(ndvi_lst_neg)
    bc_lst_neg = np.array(bc_lst_neg)
#==============================================================================
#     num_pos=int(math.ceil(len(bc_lst_pos)*pct_max))
#     num_neg=int(math.ceil(len(bc_lst_neg)*pct_max))
#==============================================================================
    num_pos=30
    num_neg=30
    #num=30   
    if len(bc_lst_pos)>0 and len(bc_lst_neg)>0:
        data_pos=nlargest(num_pos, enumerate(bc_lst_pos),key=lambda x:x[1])
        indices_pos, vals_pos = zip(*data_pos)
        data_neg=nlargest(num_neg, enumerate(abs(bc_lst_neg)),key=lambda x:x[1])
        indices_neg, vals_neg = zip(*data_neg)
        
        for ii in range(len(indices_pos)):
            ndvi_top_pos=np.append(ndvi_top_pos,ndvi_lst_pos[indices_pos[ii]])
            bc_top_pos=np.append(bc_top_pos,bc_lst_pos[indices_pos[ii]])
        for ii in range(len(indices_neg)):
            ndvi_top_neg=np.append(ndvi_top_neg,ndvi_lst_neg[indices_neg[ii]])
            bc_top_neg=np.append(bc_top_neg,bc_lst_neg[indices_neg[ii]])


#==============================================================================
# funcQuad=lambda tpl,x : tpl[0]*x**2+tpl[1]*x+tpl[2]
# func=funcQuad
# ErrorFunc=lambda tpl,x,y: func(tpl,x)-y
# tplInitial=(1.0,2.0,3.0)
# tplFinal,success=leastsq(ErrorFunc,tplInitial[:],args=(x,y))
# xx=np.linspace(x.min(),x.max(),50)
# yy=func(tplFinal,xx)
# plt.plot(xx,yy,'b-')
# plt.show()
# plt.ylim([0,10])
# print("y= %sx^2+%sx+%s"%(tplFinal[0],tplFinal[1],tplFinal[2]))
#==============================================================================

#==============================================================================
# x1=np.append(ndvi_top_pos,ndvi_top_neg)
# y1=np.append(bc_top_pos,bc_top_neg)
#==============================================================================
x=np.append(ndvi_top_pos,ndvi_top_neg)
y=np.append(bc_top_pos,bc_top_neg)
#==============================================================================
# del x,y
# x=ndvi_2
# y=bc_2
#==============================================================================
plt.plot(x,y,'go')

x=abs(x)
y=abs(y)
funcQuad=lambda tpl,x : tpl[0]*x+tpl[1]
func=funcQuad
ErrorFunc=lambda tpl,x,y: func(tpl,x)-y
tplInitial=(1.0,2.0)
tplFinal,success=leastsq(ErrorFunc,tplInitial[:],args=(x,y))
xx=np.linspace(x.min(),x.max(),50)
yy=func(tplFinal,xx)
plt.plot(xx,yy,'b-')
yy=-yy
plt.plot(xx,yy,'b-')

plt.text(0.5, 12,"y = %s x + %s"%(round(tplFinal[0],2),round(tplFinal[1],2)),
     horizontalalignment='left',
     verticalalignment='top', fontsize=15)

plt.text(0.5, -12,"y = %s x - %s"%(round(abs(tplFinal[0]),2),round(tplFinal[1],2)),
     horizontalalignment='left',
     verticalalignment='bottom', fontsize=15)

plt.ylim([-20,20])
plt.grid(True)
plt.xticks(fontsize = 15) 
plt.yticks(fontsize = 15)
plt.title("Backscatter difference between 2 adjacent days w.r.t. NDVI",fontsize=20)
plt.xlabel("NDVI",fontsize=20)
plt.ylabel("Backscatter difference",fontsize=20)

print ("y = %s x + %s"%(tplFinal[0],tplFinal[1]))
print ("y = %s x - %s"%(abs(tplFinal[0]),tplFinal[1]))
