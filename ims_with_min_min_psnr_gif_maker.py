"""

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
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
    gif_name = 'most_fluc_axial_ims_vs_t_'+pt_id if ax_cr_sg==0 else 'most_fluc_coronal_ims_vs_t_'+pt_id if ax_cr_sg==1 else 'most_fluc_sagittal_ims_vs_t_'+pt_id if ax_cr_sg==2 else 'error'
    filenames = []
    gif_ims_dir = f'/raid/yesiloglu/data/real_time_volumetric_mri/{pt_id}/temporal_evol_gifs/{gif_name}_ims'
    if not os.path.exists(gif_ims_dir):
        os.makedirs(gif_ims_dir)
    #ax_psnrs = calc_ax_psnrs(all_vols)
    cor_psnrs = calc_corr_psnrs(all_vols)
    min_psnr = cor_psnrs.min()
    max_psnr = min(cor_psnrs.max(),100)
    
    cmap = mpl.cm.get_cmap('jet_r')
    norm = mpl.colors.Normalize(vmin=min_psnr, vmax=max_psnr)
    
    plt.figure()
    plt.plot(cor_psnrs.min(0))
    plt.plot(cor_psnrs.max(0))
    #sag_psnrs = calc_sag_psnrs(all_vols)
    for t in np.arange(0, all_vols.shape[0]):
        filename = f'{gif_ims_dir}/frame_{t}.png'
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
                im_to_show = all_vols[t,:,sl_no,:]
                im = ax[i,j].imshow(im_to_show,cmap='gray', interpolation='none')
                ax[i,j].axis('off')
                ps = cor_psnrs[t,sl_no]
                ps_color = cmap((ps-min_psnr)/(max_psnr-min_psnr))
                # Create a Rectangle patch
                rect = patches.Rectangle((0, 0), im_to_show.shape[1], im_to_show.shape[0], linewidth=5, edgecolor=ps_color, facecolor='none')
                # Add the patch to the Axes
                ax[i,j].add_patch(rect)
                #print(f'max is {min_psnr}, min is {max_psnr}')
                ax[i,j].set_title('Slice {}'.format(sl_no))
                ax[i,j].text(0.5,-0.1, '(PSNR: {:.3g} dB)'.format(ps), color=ps_color, size=12, ha="center", transform=ax[i,j].transAxes)
                # color=((ps>min_psnr)*(ps<=max_psnr)*(ps-min_psnr)/(max_psnr-min_psnr) + (ps>max_psnr)*1, 0, (ps<min_psnr)*1 + (ps>=min_psnr)*(ps<max_psnr)*(max_psnr-ps)/(max_psnr-min_psnr)))
                divider = make_axes_locatable(ax[i,j])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im, cax=cax, orientation='vertical')
        plt.subplots_adjust(left=0.01, right=0.90, bottom=0.05, top=0.935, wspace=0.32)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
        
        cb1 = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap,
                                        norm=norm,
                                        orientation='vertical')
        cb1.set_label('PSNR wrt the Initial Image (dB)')
        
        #fig.show()
        
        plt.show()
        if ax_cr_sg == 0:
            plt.suptitle(f"Most Fluctuating Axial Images ({pt_id}, Time Point: {t})")
        elif ax_cr_sg == 1:
            plt.suptitle(f"Most Fluctuating Coronal Images ({pt_id}, Time Point: {t})")
        elif ax_cr_sg == 2:
            plt.suptitle(f"Most Fluctuating Sagittal Images ({pt_id}, Time Point: {t})")
        plt.savefig(filename, dpi=96, bbox_inches='tight')
        plt.close()
    print('Charts saved\n')
    # Build GIF
    print('Creating gif\n')
    with imageio.get_writer(f'/raid/yesiloglu/data/real_time_volumetric_mri/{pt_id}/temporal_evol_gifs/{gif_name}.gif', mode='I') as writer:
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