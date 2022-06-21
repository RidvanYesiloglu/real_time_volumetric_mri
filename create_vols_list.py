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
# Imports
import os

# Methods
def extract_vol_no(name):
    return int(name[name.find('_')+1:name.find('.')])
def pts_str_from_id_list(pt_ids):
    pts_str = ''
    for no, pt_id in enumerate(pt_ids):
        pts_str += '{} ({})\n'.format(pt_id, no)
    return pts_str

# Constants
all_data_folder = '/raid/yesiloglu/data/real_time_volumetric_mri/'
pt_ids = ['patient19', 'patient73', 'pt_19_5min', 'pt_56_5min', 'pt_73_5min', \
               'pt_82_5min', 'pt_85_5min', 'pt_92_5min', 'pt_95_5min']
# Get patient id from user:
pt_id = pt_ids[input('Patient ids are:\n{}\nChoose patient (use the numbers inside par.): '.format(pts_str_from_id_list(pt_ids)))]
# Create volumes list:
patient_data_folder = all_data_folder + pt_id
vol_list = os.listdir(patient_data_folder)
print('Vol list before sorting: ')
print(vol_list)
vol_list.sort(key=extract_vol_no)
print('Vol list after sorting: ')
print(vol_list)