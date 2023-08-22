# -*- coding: utf-8 -*-
"""
Created on Wed Jun 07 14:50:36 2017

@author: Qi
"""
import glob
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
import pandas as pd
import gdal
from datetime import timedelta
#==============================================================================
# plot Ragola
#==============================================================================

        
SM_retrieved_Ragola=np.load('D:/01_Projects/01_EO_Africa/Data/npy/SM_retrieved_Ragola_19classes.npy', allow_pickle=True)
SM_insitu_Ragola=np.load('D:/01_Projects/01_EO_Africa/Data/npy/SM_insitu_Ragola_19classes.npy', allow_pickle=True)
date_Ragola=np.load('D:/01_Projects/01_EO_Africa/Data/npy/date_Ragola_19classes.npy', allow_pickle=True)

Mv_diff_A=np.zeros((len(date_Ragola),1))
for i in range(len(date_Ragola)):
    if date_Ragola[i]-date_Ragola[i-1]<timedelta(13):
        if not np.isnan(Mv_diff_A[i]):
            Mv_diff_A[i]=SM_retrieved_Ragola[i]-SM_retrieved_Ragola[i-1]
Mv_diff_A[Mv_diff_A==np.nan]=0        
print('%s'%max(Mv_diff_A))     
