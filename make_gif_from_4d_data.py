"""

"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import imageio
import os

def main(all_vols, pt_id):
    print('this is main method')
    gif_name = 'axials_vs_t_'+pt_id
    filenames = []
    temp_dir = f'/raid/yesiloglu/data/real_time_volumetric_mri/{pt_id}/{gif_name}_ims'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    for t in np.arange(0, all_vols.shape[0]):
        filename = f'{temp_dir}/frame_{t}.png'
        filenames.append(filename)
        
        # last frame of each viz stays longer
        if (t == (all_vols.shape[0]-1)):
            for t in range(5):
                filenames.append(filename)
                
        # save img
        fig,ax = plt.subplots(4,8, figsize=(16,8.66))
        for i in range(4):
            for j in range(8):
                sl_no = (8*i+j)*2
                im = ax[i,j].imshow(all_vols[t,...,sl_no],cmap='gray', interpolation='none')
                ax[i,j].axis('off')
                ax[i,j].set_title(f'Slice: {sl_no}')
                divider = make_axes_locatable(ax[i,j])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im, cax=cax, orientation='vertical')
        plt.subplots_adjust(left=0, right=0.97, bottom=0, top=0.935, wspace=0.32)
        # manager = plt.get_current_fig_manager()
        # manager.full_screen_toggle()
        plt.show()
        plt.suptitle(f"Axial Images ({pt_id}, Time Point: {t})")
        plt.savefig(filename, dpi=96, bbox_inches='tight')
        size = fig.get_size_inches() # size in inches
        print('Size is ', size)
        #plt.close()
    print('Charts saved\n')
    # Build GIF
    print('Creating gif\n')
    with imageio.get_writer(f'/raid/yesiloglu/data/real_time_volumetric_mri/{pt_id}/{gif_name}.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    print('Gif saved\n')
    #print('Removing Images\n')
    # Remove files
    # for filename in set(filenames):
    #     os.remove(filename)
    print('DONE')
    
    
if __name__ == "__main__":
    main()