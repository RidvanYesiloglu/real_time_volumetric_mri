"""

"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import imageio
import os
def calc_ax_psnrs(ims):
    return -10*np.log(((ims[0:1]-ims)**2).mean((1,2)))
def calc_corr_psnrs(ims):
    return -10*np.log(((ims[0:1]-ims)**2).mean((1,3)))
def calc_sag_psnrs(ims):
    return 20*np.log((ims[0:1]**2).max((2,3)))-10*np.log(((ims[0:1]-ims)**2).mean((2,3)))
def main(all_vols, pt_id, ax_cr_sg):
    print('this is main method')
    gif_name = 'dist_axial_ims_vs_t_'+pt_id if ax_cr_sg==0 else 'dist_coronal_ims_vs_t_'+pt_id if ax_cr_sg==1 else 'dist_sagittal_ims_vs_t_'+pt_id if ax_cr_sg==2 else 'error'
    filenames = []
    temp_dir = f'/raid/yesiloglu/data/real_time_volumetric_mri/{pt_id}/{gif_name}_ims'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    #ax_psnrs = calc_ax_psnrs(all_vols)
    cor_psnrs = calc_corr_psnrs(all_vols)
    min_psnr = cor_psnrs.min()
    max_psnr = min(cor_psnrs.max(),100)
    
    plt.figure()
    plt.plot(cor_psnrs.min(0))
    plt.plot(cor_psnrs.max(0))
    #sag_psnrs = calc_sag_psnrs(all_vols)
    for t in np.arange(0, all_vols.shape[0]):
        filename = f'{temp_dir}/frame_{t}.png'
        filenames.append(filename)
        
        # last frame of each viz stays longer
        if (t == (all_vols.shape[0]-1)):
            for t in range(5):
                filenames.append(filename)
                
        # save img
        sl_nos = [41,42,43,44,45,46,47,48,49,50,51,52,64,72,80,88]
        fig,ax = plt.subplots(2,8, figsize=(16,8.66))
        for i in range(2):
            for j in range(8):
                sl_no = sl_nos[8*i+j]
                im = ax[i,j].imshow(all_vols[t,:,sl_no,:],cmap='gray', interpolation='none')
                ax[i,j].axis('off')
                ps = cor_psnrs[t,sl_no]
                
                #print(f'max is {min_psnr}, min is {max_psnr}')
                ax[i,j].set_title('Slice {}\n(PSNR {:.3g})'.format(sl_no, ps), color=((ps>min_psnr)*(ps<=max_psnr)*(ps-min_psnr)/(max_psnr-min_psnr) + (ps>max_psnr)*1, 0, (ps<min_psnr)*1 + (ps>=min_psnr)*(ps<max_psnr)*(max_psnr-ps)/(max_psnr-min_psnr)))
                divider = make_axes_locatable(ax[i,j])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im, cax=cax, orientation='vertical')
        plt.subplots_adjust(left=0, right=0.97, bottom=0.05, top=0.935, wspace=0.32)
        plt.show()
        if ax_cr_sg == 0:
            plt.suptitle(f"Most Distinctive Axial Images ({pt_id}, Time Point: {t})")
        elif ax_cr_sg == 1:
            plt.suptitle(f"Most Distinctive Coronal Images ({pt_id}, Time Point: {t})")
        elif ax_cr_sg == 2:
            plt.suptitle(f"Most Distinctive Sagittal Images ({pt_id}, Time Point: {t})")
        plt.savefig(filename, dpi=96, bbox_inches='tight')
        plt.close()
    print('Charts saved\n')
    # Build GIF
    print('Creating gif\n')
    with imageio.get_writer(f'/raid/yesiloglu/data/real_time_volumetric_mri/{pt_id}/{gif_name}.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    print('Gif saved\n')
    print('Removing Images\n')
    #Remove files
    for filename in set(filenames):
        os.remove(filename)
    print('DONE')
    
    
if __name__ == "__main__":
    main()