"""
• Creates the list of the names of the bin files inside a patient's data folder.
• Assumes that data is located on /raid/yesiloglu/data/real_time_volumetric_mri/'x' 
where 'x' is one of the following patient directory names:
    o	patient19
    o	patient73
    o	pt_19_5min
    o	pt_56_5min
    o	pt_73_5min
    o	pt_82_5min
    o	pt_85_5min
    o	pt_92_5min
    o	pt_95_5min
• Data is obtained at a temporal sampling rate of 340 ms.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from mpl_toolkits.axes_grid1 import make_axes_locatable

def normalize(img, min_val=None, max_val=None):
    min_val = min_val if min_val is not None else img.min()
    max_val = max_val if max_val is not None else img.max()
    return (img-min_val)/(max_val-min_val)
def extract_vol_no(name):
    return int(name[name.find('_')+1:name.find('.')])
def psnr_arr(ims):
    return -10*np.log(((ims[0:1]-ims[1:])**2).mean((1,2,3)))

data_folder = os.path.join('data_4DMRI',dtname) # 'patient19'
dtname = 'pt_95_5min'
patient_list = os.listdir(data_folder)
patient_list.sort(key=extract_vol_no)
def create_patient_list():
    return 1