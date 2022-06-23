"""

"""
import numpy as np
import matplotlib.pyplot as plt
import create_vol_dirs_list
import make_gif_from_4d_data
import special_gif_maker
def main(pt_id=None):
    # Constants
    all_data_folder = '/raid/yesiloglu/data/real_time_volumetric_mri/'
    #all_data_folder = 'C:/Users/ridva/OneDrive/Masaüstü/XING Lab/data/real_time_volumetric_mri'
    vol_size = (128,128,64)
    pt_ids = ['patient19', 'patient73', 'pt_19_5min', 'pt_56_5min', 'pt_73_5min', \
                   'pt_82_5min', 'pt_85_5min', 'pt_92_5min', 'pt_95_5min']
    for pt_id in pt_ids:
        if ((pt_id == 'patient19') or (pt_id == 'patient73')):
            continue
        vol_dirs = create_vol_dirs_list.main(pt_id, sort=True)
        
        data_4d = np.zeros((len(vol_dirs),) + vol_size)
        for vol_no, vol_dir in enumerate(vol_dirs):
            img = np.fromfile(vol_dir, dtype='float32')
            data_4d[vol_no] = np.reshape(img.transpose(), vol_size, order="F")
        
        #np.save(all_data_folder+'/'+pt_id+'/all_vols',data_4d)
        #make_gif_from_4d_data.main(data_4d, pt_id, 1)
        #make_gif_from_4d_data.main(data_4d, pt_id, 2)
        special_gif_maker.main(data_4d, pt_id, 1)
if __name__ == "__main__":
    main()