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
import numpy as np
# Methods
def extract_vol_no(vol_name):
    # Returns the volume number from volume string
    # Example: 'volume_132.bin' => 132
    return int(vol_name[vol_name.find('_')+1:vol_name.find('.')])

def pts_str_from_id_list(pt_ids):
    # Create a string listing the patient ids like:
    # patient19 (0)
    # patient73 (1) ...
    pts_str = ''
    for no, pt_id in enumerate(pt_ids):
        pts_str += '{} ({})\n'.format(pt_id, no)
    return pts_str

def main():
    # Constants
    all_data_folder = '/raid/yesiloglu/data/real_time_volumetric_mri/'
    pt_ids = ['patient19', 'patient73', 'pt_19_5min', 'pt_56_5min', 'pt_73_5min', \
                   'pt_82_5min', 'pt_85_5min', 'pt_92_5min', 'pt_95_5min']
    # Get patient id from user:
    pt_id = pt_ids[int(input('Patient ids are:\n{}\nChoose patient (use the numbers inside par.): '.format(pts_str_from_id_list(pt_ids))))]
    # Create volumes list:
    patient_data_folder = all_data_folder + pt_id
    print('Data folder was set to {}'.format(patient_data_folder))
    vol_list = os.listdir(patient_data_folder)
    # Sort volumes list according to volume no
    vol_list.sort(key=extract_vol_no)
    # Print min, max, length etc. for the array of volume nos:
    vol_nos = np.asarray([extract_vol_no(no) for no in vol_list])
    print('Len is {}, min is {}, max is {}'.format(vol_nos.size, vol_nos.min(), vol_nos.max()))    
    print('Min adj difference is {}, max adj difference is {}'.format(np.diff(vol_nos).min(), np.diff(vol_nos).max()))
    # Return volumes list:
    return vol_list
if __name__ == "__main__":
    main()