"""

"""
import numpy as np
import create_vol_dirs_list
import make_gif_from_4d_data
import make_3dpsnr_wrti_pl_from_4d_data
import multiprocessing
import time

#import special_gif_maker
def main(pt_id=None):
    # Constants
    all_data_folder = '/raid/yesiloglu/data/real_time_volumetric_mri/'
    #all_data_folder = 'C:/Users/ridva/OneDrive/Masaüstü/XING Lab/data/real_time_volumetric_mri'
    vol_size = (128,128,64)
    pt_ids = ['patient19', 'patient73', 'pt_19_5min', 'pt_56_5min', 'pt_73_5min', \
                   'pt_82_5min', 'pt_85_5min', 'pt_92_5min', 'pt_95_5min']
    for pt_id in pt_ids:
        print(f'Now, patient {pt_id} is being processed.')
        start_time = time.perf_counter()
        if ((pt_id == 'patient19') or (pt_id == 'patient73')):
            print(f'Patient {pt_id} skipped.')
            continue
        
        print('List of .vol files for the patient is being created.')
        vol_dirs = create_vol_dirs_list.main(pt_id, sort=True)
        finish_time = time.perf_counter()
        print(f"Elapsed total for the patient: {finish_time-start_time} seconds")
        print('The vol files'' for the patient are being merged in a numpy array.')
        all_vols = np.zeros((len(vol_dirs),) + vol_size)
        for vol_no, vol_dir in enumerate(vol_dirs):
            img = np.fromfile(vol_dir, dtype='float32')
            all_vols[vol_no] = np.reshape(img.transpose(), vol_size, order="F")
        finish_time = time.perf_counter()
        #all_vols = np.load(all_data_folder+'/'+pt_id+'/all_vols.npy')
        np.save(all_data_folder+'/'+pt_id+'/all_vols',all_vols)
        print(f"Elapsed total for the patient: {finish_time-start_time} seconds")
        print('3D PSNRs wrt the initial image are being plotted.')
        make_3dpsnr_wrti_pl_from_4d_data.main(all_vols)
        #print('The gifs for the data is being plotted and saved.')
        # processes = []
        # for plot_most_fluc in [False, True]:
        #     for ax_cr_sg in [0,1,2]:
        #         p = multiprocessing.Process(target = make_gif_from_4d_data.main, args=(all_vols, pt_id, ax_cr_sg, plot_most_fluc,))
        #         p.start()
        #         processes.append(p)
        # # Join all the processes 
        # for p in processes:
        #     p.join()
        finish_time = time.perf_counter()
        print(f"Elapsed total for the patient: {finish_time-start_time} seconds, done.")
if __name__ == "__main__":
    main()