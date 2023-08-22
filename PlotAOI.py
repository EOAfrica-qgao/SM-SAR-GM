# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 15:36:44 2022

@author: qgao
"""


import glob
import numpy as np
from matplotlib import pyplot as plt
import gdal
from pykml import parser
from astropy.visualization import make_lupton_rgb
from matplotlib.ticker import FormatStrFormatter
# import rasterio

def normalize(band):
    band_min, band_max = (band.min(), band.max())
    return ((band-band_min)/((band_max - band_min)))
def gammacorr(band):
    gamma=50
    return np.power(band, 1/gamma)


AOI_mask=("D:/01_Projects/01_EO_Africa/Data/20201029T113321_20201029T113454_T28PCB.tif")

dataset = gdal.Open(AOI_mask)
gt = dataset.GetGeoTransform()
red = dataset.GetRasterBand(4).ReadAsArray()
green = dataset.GetRasterBand(3).ReadAsArray()
blue = dataset.GetRasterBand(2).ReadAsArray()
projection = dataset.GetProjection()


red_g=gammacorr(red)
blue_g=gammacorr(blue)
green_g=gammacorr(green)

red_gn = normalize(red_g)
green_gn = normalize(green_g)
blue_gn = normalize(blue_g)

rgb_composite_n = np.dstack((red_gn, green_gn, blue_gn)).transpose(0, 1, 2)




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
plt.figure()
plt.subplot(1, 1, 1)
img = plt.imshow(rgb_composite_n, extent=extent,origin='upper')
ax = plt.gca()
ax.get_yaxis().get_major_formatter().set_useOffset(False) 
plt.xticks(fontsize = 10) 
plt.yticks(fontsize = 10)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#==============================================================================
# plt.rc('xtick', labelsize=16)      
# plt.rc('ytick', labelsize=16) 
#==============================================================================

# plt.axis('tight')
plt.xlabel("Longitude [deg]",fontsize=10)
plt.ylabel("Latitude [deg]",fontsize=10)

plt.savefig("D:/01_Projects/01_EO_Africa/AOI.png",dpi=1500)