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
def extract_vol_no(vol_name):
    # Returns the volume number from volume string
    # Example: 'volume_132.bin' => 132
    return int(vol_name[vol_name.find('_')+1:vol_name.find('.')])

def pts_str_from_id_list(pt_ids):
    pts_str = ''
    for no, pt_id in enumerate(pt_ids):
        pts_str += '{} ({})\n'.format(pt_id, no)
    return pts_str

def ana_vol_nos(sorted_vol_list):
    starts = [extract_vol_no(sorted_vol_list[0])]
    ends = []
    prev_vol_no = extract_vol_no(sorted_vol_list[0])
    for vol_name in sorted_vol_list[1:]:
        vol_no = extract_vol_no(vol_name)
        if (vol_no != (prev_vol_no + 1)):
            ends.append(prev_vol_no)
            starts.append(vol_no)
        prev_vol_no = vol_no
    ends.append(vol_no)
    return ['{}:{}, '.format(starts[ind], ends[ind]) for ind in range(len(starts))]
        
    
# Constants
all_data_folder = '/raid/yesiloglu/data/real_time_volumetric_mri/'
pt_ids = ['patient19', 'patient73', 'pt_19_5min', 'pt_56_5min', 'pt_73_5min', \
               'pt_82_5min', 'pt_85_5min', 'pt_92_5min', 'pt_95_5min']
# Get patient id from user:
pt_id = pt_ids[int(input('Patient ids are:\n{}\nChoose patient (use the numbers inside par.): '.format(pts_str_from_id_list(pt_ids))))]
# Create volumes list:
patient_data_folder = all_data_folder + pt_id
vol_list = os.listdir(patient_data_folder)
#print('Vol list before sorting: ')
#print(vol_list)
vol_list.sort(key=extract_vol_no)
#print('Vol list after sorting: ')
#print(vol_list)
print(ana_vol_nos(vol_list))