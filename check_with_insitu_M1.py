# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 12:10:38 2023

@author: qgao
"""


import glob
import numpy as np
import gdal
import os
from datetime import datetime
from datetime import timedelta
from matplotlib import pyplot as plt
import pandas as pd
from scipy.optimize import leastsq
import csv
from scipy import stats
from scipy.stats import pearsonr

#%%

track=str('133')
hh=19
mm=18

    
field=str('target_field')

ndvi_class=np.zeros((20,1))*np.nan
for i in range(20):
    ndvi_class[i]=[0.1*i/2]
bc_min = np.load('D:/01_Projects/01_EO_Africa/Data/npy/bc_min_s2_ndvi_19classes.npy')
ndvi = np.load('D:/01_Projects/01_EO_Africa/Data/npy/s2_ndvi_%s.npy'%track)

def f_ndvi(ndvi):
    # data = -19.28 *ndvi + 14.92 #filtered
    data = -23.197606021892618 *ndvi + 17.63255650873604
    return data
    
# =============================================================================
# mv_min=0.133 #field
# Mv_max=0.33
# =============================================================================
Mv_min=0.0 #field
Mv_max=0.3

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

field_mask=("D:/01_Projects/01_EO_Africa/Ragola.tif")
dataset = gdal.Open(field_mask)
geotransform = dataset.GetGeoTransform()
field = dataset.GetRasterBand(1).ReadAsArray()
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
Mv_field=np.zeros((n,1))*np.nan
date_bc=np.empty((n,0)).tolist()
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
    
    
    Mv_lst=[]
    for x in range(200):
        for y in range(200):
            if not np.isnan(ndvi[x,y,i]):
                av_bc=np.nan
                bc_minimum=np.nan
                if mask_field[x,y]==1:  
                    for p in range(19):
                        if ndvi[x,y,i]>ndvi_class[p] and ndvi[x,y,i]<ndvi_class[p+1]:
                            bc_minimum=bc_min[x,y,p]
                    if not np.isnan(bc_minimum):
                        xl=x*10
                        xr=(x+1)*10
                        yb=y*10
                        yu=(y+1)*10
            
                        bc_win=bc[xl:xr,yb:yu]
                        
                        av_bc=np.mean(bc_win)
                        if not np.isnan(av_bc):
                            av_bc=linear2db(av_bc)
                
                        f=f_ndvi(ndvi[x,y,i])
                        
                        bc_diff=av_bc-bc_minimum
                        if bc_diff>f:
                            bc_diff=f
                        Mv=(av_bc-bc_minimum)/f*Mv_max+Mv_min
                        if Mv<Mv_min:
                            Mv=Mv_min
                        if Mv>Mv_max:
                            Mv=Mv_max
                        Mv_lst=np.append(Mv_lst,Mv)
    Mv_field[i]=np.nanmean(Mv_lst)
    

#%%
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()  



#%%

csv_file_path = ("D:/01_Projects/01_EO_Africa/Data/Insitu_data/AMMA-CATCH/PA.SW_SNNr_BD_AMMA-CATCH_2023_03_20/PA.SW_SNNr-RAGOLA-2018.csv")

# Read the CSV file into a pandas DataFrame, starting from row 31 (index 30)
df = pd.read_csv(csv_file_path, skiprows=range(1, 30),sep=';')

dates = []
values = []
for index, row in df.iterrows():
    date = row[0]
    value = row[1]
    dates.append(date)
    values.append(value)

date_format = '%Y-%m-%d %H:%M:%S'
date_times_Ragola_2018 = [datetime.strptime(date_str, date_format) for date_str in dates]
sm_values_2018 = [num * 0.01 for num in values]

            
leg2,=ax1.plot(date_times_Ragola_2018,sm_values_2018,color="lime",marker='.')
ax1.set_ylabel('Soil Moisture', fontsize=20)

csv_file_path = ("D:/01_Projects/01_EO_Africa/Data/Insitu_data/AMMA-CATCH/PA.SW_SNNr_BD_AMMA-CATCH_2023_03_20/PA.SW_SNNr-RAGOLA-2019.csv")

# Read the CSV file into a pandas DataFrame, starting from row 31 (index 30)
df = pd.read_csv(csv_file_path, skiprows=range(1, 30),sep=';')

dates = []
values = []
for index, row in df.iterrows():
    date = row[0]
    value = row[1]
    dates.append(date)
    values.append(value)

date_format = '%Y-%m-%d %H:%M:%S'
date_times_Ragola_2019 = [datetime.strptime(date_str, date_format) for date_str in dates]
sm_values_2019 = [num * 0.01 for num in values]            
ax1.plot(date_times_Ragola_2019,sm_values_2019,color="lime",marker='.')


csv_file_path = ("D:/01_Projects/01_EO_Africa/Data/Insitu_data/AMMA-CATCH/PA.SW_SNNr_BD_AMMA-CATCH_2023_03_20/PA.SW_SNNr-RAGOLA-2020.csv")

# Read the CSV file into a pandas DataFrame, starting from row 31 (index 30)
df = pd.read_csv(csv_file_path, skiprows=range(1, 30),sep=';')

dates = []
values = []
for index, row in df.iterrows():
    date = row[0]
    value = row[1]
    dates.append(date)
    values.append(value)

date_format = '%Y-%m-%d %H:%M:%S'
date_times_Ragola_2020 = [datetime.strptime(date_str, date_format) for date_str in dates]
sm_values_2020 = [num * 0.01 for num in values]            
ax1.plot(date_times_Ragola_2020,sm_values_2020,color="lime",marker='.')


csv_file_path = ("D:/01_Projects/01_EO_Africa/Data/Insitu_data/AMMA-CATCH/PA.SW_SNNr_BD_AMMA-CATCH_2023_03_20/PA.SW_SNNr-RAGOLA-2021.csv")

# Read the CSV file into a pandas DataFrame, starting from row 31 (index 30)
df = pd.read_csv(csv_file_path, skiprows=range(1, 30),sep=';')

dates = []
values = []
for index, row in df.iterrows():
    date = row[0]
    value = row[1]
    dates.append(date)
    values.append(value)

date_format = '%Y-%m-%d %H:%M:%S'
date_times_Ragola_2021 = [datetime.strptime(date_str, date_format) for date_str in dates]
sm_values_2021 = [num * 0.01 for num in values]          
ax1.plot(date_times_Ragola_2021,sm_values_2021,color="lime",marker='.')


date_times_Ragola = date_times_Ragola_2018 + date_times_Ragola_2019 +date_times_Ragola_2020 +date_times_Ragola_2021
insitu_sm_values = sm_values_2018 + sm_values_2019 + sm_values_2020 + sm_values_2021
# =============================================================================
# plt.xlim([datetime(2017,3,1),datetime(2018,7,20)])
# =============================================================================
# =============================================================================
# plt.gcf().autofmt_xdate()
# =============================================================================

#%% plot precipattaion


csv_file_path = ("D:/01_Projects/01_EO_Africa/Data/Insitu_data/AMMA-CATCH/PA.Met_SNNs_BD_AMMA-CATCH_2023_07_07/PA.Met_SNNs-FAIDHERBIA-FLUX-2018.csv")

# Read the CSV file into a pandas DataFrame, starting from row 31 (index 30)
df = pd.read_csv(csv_file_path, skiprows=range(1, 30),sep=';')

dates = []
values = []
for index, row in df.iterrows():
    date = row[0]
    value = row[1]
    dates.append(date)
    values.append(value)
 

date_format = '%Y-%m-%d %H:%M:%S'
date_times_meteo = [datetime.strptime(date_str, date_format) for date_str in dates]

for i in range(len(values)):
    if values[i]<=0:
        values[i]=np.nan
pre_values = [num for num in values] 
width = 1.5
ax2.bar(date_times_meteo,pre_values, width, color="blue")

csv_file_path = ("D:/01_Projects/01_EO_Africa/Data/Insitu_data/AMMA-CATCH/PA.Met_SNNs_BD_AMMA-CATCH_2023_07_07/PA.Met_SNNs-FAIDHERBIA-FLUX-2019.csv")

# Read the CSV file into a pandas DataFrame, starting from row 31 (index 30)
df = pd.read_csv(csv_file_path, skiprows=range(1, 30),sep=';')

dates = []
values = []
for index, row in df.iterrows():
    date = row[0]
    value = row[1]
    dates.append(date)
    values.append(value)
 

date_format = '%Y-%m-%d %H:%M:%S'
date_times_meteo = [datetime.strptime(date_str, date_format) for date_str in dates]

for i in range(len(values)):
    if values[i]<=0:
        values[i]=np.nan
pre_values = [num for num in values] 
width = 1.5
ax2.bar(date_times_meteo,pre_values, width, color="blue")

csv_file_path = ("D:/01_Projects/01_EO_Africa/Data/Insitu_data/AMMA-CATCH/PA.Met_SNNs_BD_AMMA-CATCH_2023_07_07/PA.Met_SNNs-FAIDHERBIA-FLUX-2020.csv")

# Read the CSV file into a pandas DataFrame, starting from row 31 (index 30)
df = pd.read_csv(csv_file_path, skiprows=range(1, 30),sep=';')

dates = []
values = []
for index, row in df.iterrows():
    date = row[0]
    value = row[1]
    dates.append(date)
    values.append(value)
 

date_format = '%Y-%m-%d %H:%M:%S'
date_times_meteo = [datetime.strptime(date_str, date_format) for date_str in dates]

for i in range(len(values)):
    if values[i]<=0:
        values[i]=np.nan
pre_values = [num for num in values] 
width = 1.5
ax2.bar(date_times_meteo,pre_values, width, color="blue")

csv_file_path = ("D:/01_Projects/01_EO_Africa/Data/Insitu_data/AMMA-CATCH/PA.Met_SNNs_BD_AMMA-CATCH_2023_07_07/PA.Met_SNNs-FAIDHERBIA-FLUX-2021.csv")

# Read the CSV file into a pandas DataFrame, starting from row 31 (index 30)
df = pd.read_csv(csv_file_path, skiprows=range(1, 30),sep=';')

dates = []
values = []
for index, row in df.iterrows():
    date = row[0]
    value = row[1]
    dates.append(date)
    values.append(value)
 

date_format = '%Y-%m-%d %H:%M:%S'
date_times_meteo = [datetime.strptime(date_str, date_format) for date_str in dates]

for i in range(len(values)):
    if values[i]<=0:
        values[i]=np.nan
pre_values = [num for num in values] 
width = 1.5
ax2.bar(date_times_meteo,pre_values, width, color="blue")


ax2.set_ylabel('Precipitation', fontsize=20)    
ax2.yaxis.set_tick_params(labelsize=15)
plt.xlim([datetime(2018,1,1),datetime(2021,4,1)])

#%%
n= len(bc_files)
Mv_av_field=np.empty(n)
for i in range(n):
    Mv_av_field[i]=Mv_field[i]
ax1.plot(date_bc,Mv_av_field,'ro')
Mv_i=[]
date_i=[]
for i in range(n):
    if not np.isnan(Mv_av_field[i]):
       Mv_i=np.append(Mv_i,Mv_av_field[i])
       date_i=np.append(date_i,date_bc[i])
leg1,=ax1.plot(date_i,Mv_i,'r')
plt.title("Time series of Soil Moisture Retrieved in target field ",fontsize=20)


ax1.yaxis.set_tick_params(labelsize=15)
ax1.xaxis.set_tick_params(labelsize=15, rotation=45)
plt.legend([leg1, leg2], ['Retrieved SM (S1)', 'in-situ SM'],loc='upper right')



#%%

nn= len(date_i)
si_m=np.zeros(nn)*np.nan
jj=len(date_times_Ragola)
#==============================================================================
# N_AB42015=np.zeros(jj2015)*np.nan
# N_AB42016=np.zeros(jj2016)*np.nan
# N_bc=np.zeros(nn)*np.nan
# for j in xrange(jj2015):
#     N_AB42015[j]=np.int(datetime.toordinal(datetime.date(SM_AB4_date2015[j])))
# for j in xrange(jj2016):
#     N_AB42016[j]=np.int(datetime.toordinal(datetime.date(SM_AB4_date2016[j])))
# for i in xrange(nn):
#     N_bc[i]=np.int(datetime.toordinal(date_i[i]))
#==============================================================================
    
  
for i in range(nn):
    si=[]
    for j in range(jj):
        if abs(date_times_Ragola[j]-date_i[i])<timedelta(0, 100):
            si=np.append(si,insitu_sm_values[j])
        si_m[i]=np.mean(si)
        #if N_AB42015[j]==N_bc[i]:
            #si=np.append(si,SM_AB4_2015[j])
    #si_m[i]=np.mean(si)


    
plt.figure()
plt.plot(si_m,Mv_i,'ro')

plt.ylabel('Retrieved Soil Moilsture', fontsize=20)
plt.xlabel('In-situ Soil Moilsture', fontsize=20)
plt.title("Soil Moisture in Ragola ",fontsize=20)

#==============================================================================
#     for i in xrange(len(date_i)):
#         if date_i[i]<datetime(2016,1,1):
#             Mv_i[i]=np.nan
#==============================================================================
np.save('D:/01_Projects/01_EO_Africa/Data/npy/SM_retrieved_Ragola_19classes.npy',Mv_i)
np.save('D:/01_Projects/01_EO_Africa/Data/npy/SM_insitu_Ragola_19classes.npy',si_m)
np.save('D:/01_Projects/01_EO_Africa/Data/npy/date_Ragola_19classes.npy',date_i)
    
#%%
SM_retrieved_Ragola=np.load('D:/01_Projects/01_EO_Africa/Data/npy/SM_retrieved_Ragola_19classes.npy', allow_pickle=True)
SM_insitu_Ragola=np.load('D:/01_Projects/01_EO_Africa/Data/npy/SM_insitu_Ragola_19classes.npy', allow_pickle=True)
date_Ragola=np.load('D:/01_Projects/01_EO_Africa/Data/npy/date_Ragola_19classes.npy', allow_pickle=True)
#==============================================================================
# SM_retrieved_AB4_30=np.load('C:/Users/Qi/Desktop/SM_QI/code 100m/npy/SM_retrieved_AB4_9classes_30_60km_cut.npy')
# SM_insitu_AB4_30=np.load('C:/Users/Qi/Desktop/SM_QI/code 100m/npy/SM_insitu_AB4_9classes_30_60km_cut.npy')
# 
# SM_retrieved_AB4_132=np.load('C:/Users/Qi/Desktop/SM_QI/code 100m/npy/SM_retrieved_AB4_9classes_132_60km_cut.npy')
# SM_insitu_AB4_132=np.load('C:/Users/Qi/Desktop/SM_QI/code 100m/npy/SM_insitu_AB4_9classes_132_60km_cut.npy')
#==============================================================================

#==============================================================================
# SM_insitu_AB4_110[1:10]=np.nan
# SM_insitu_AB4_30[1:10]=np.nan
# SM_insitu_AB4_132[1:10]=np.nan
#==============================================================================

plt.figure()
plt.plot(SM_insitu_Ragola,SM_retrieved_Ragola,'ro')
#==============================================================================
# plt.plot(SM_insitu_AB4_30,SM_retrieved_AB4_30,'go')
# plt.plot(SM_insitu_AB4_132,SM_retrieved_AB4_132,'bo')
#==============================================================================
SM_retrieved=SM_retrieved_Ragola
SM_insitu=SM_insitu_Ragola
#==============================================================================
# SM_retrieved=np.concatenate((SM_retrieved_AB4_110, SM_retrieved_AB4_30,SM_retrieved_AB4_132))
# SM_insitu=np.concatenate((SM_insitu_AB4_110, SM_insitu_AB4_30,SM_insitu_AB4_132))
#==============================================================================


plt.xlabel('In-situ Soil Moilsture', fontsize=20)
plt.ylabel('Retrieved Soil Moilsture', fontsize=20)
plt.title("Soil Moisture in Ragola Field",fontsize=20)
plt.ylim(0,0.25)
plt.xlim(0,0.25)


x=[]
y=[]
n=len(SM_insitu)
for i in range(n):
    if not np.isnan(SM_insitu[i]):
        x=np.append(x,SM_insitu[i])
        y=np.append(y,SM_retrieved[i])
xx=x
yy=x
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
def bias(predictions, targets):
    return (predictions - targets).mean()

def ub_rmse(predictions, targets,biases):
    return np.sqrt(((predictions-targets-biases)**2).mean())   
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
r_squared=r_value**2
  
rms=rmse(yy,y)
biases=bias(yy,y)
ubrms=ub_rmse(yy,y,biases)


plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.plot([0,0.1,0.2,0.3],[0,0.1,0.2,0.3])
#plt.plot(xx,yy,'r-')
plt.text(0.02, 0.23,'RMSE=%s\nBias=%s\nubRMSE=%s\nR$^2$=%s'%(round(rms,3),round(biases,3),round(ubrms,3),round(r_squared,3)),
     horizontalalignment='left',
     verticalalignment='top', fontsize=15)
