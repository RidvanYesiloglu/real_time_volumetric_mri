"""
Does for a specific patient either
- all_axials_vs_t_
- most_fluc_axials_vs_t_
- all_coronals_vs_t_
- most_fluc_coronals_vs_t_
- all_sagittals_vs_t_
- most_fluc_sagittals_vs_t_

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import imageio
import os

def calc_psnrs(all_vols, ax_cr_sg):
    if ax_cr_sg == 0:
        return -10*np.log(((all_vols[0:1]-all_vols)**2).mean((1,2)))
    elif ax_cr_sg == 1:
        return -10*np.log(((all_vols[0:1]-all_vols)**2).mean((1,3)))
    elif ax_cr_sg == 2:
        return 20*np.log((all_vols[0:1]**2).max((2,3)))-10*np.log(((all_vols[0:1]-all_vols)**2).mean((2,3)))
    
def main(all_vols, pt_id, ax_cr_sg, plot_most_fluc=False):
    print('this is main method')
    if plot_most_fluc:
        gif_name = 'most_fluc_axials_vs_t_'+pt_id if ax_cr_sg==0 else 'most_fluc_coronals_vs_t_'+pt_id if ax_cr_sg==1 else 'most_fluc_sagittals_vs_t_'+pt_id if ax_cr_sg==2 else 'error'
    else:
        gif_name = 'all_axials_vs_t_'+pt_id if ax_cr_sg==0 else 'all_coronals_vs_t_'+pt_id if ax_cr_sg==1 else 'all_sagittals_vs_t_'+pt_id if ax_cr_sg==2 else 'error'
    ind_ims_dir = f'/raid/yesiloglu/data/real_time_volumetric_mri/{pt_id}/temporal_evol_gifs/{gif_name}_ims'    
    if not os.path.exists(ind_ims_dir):
        os.makedirs(ind_ims_dir)
        
    psnrs = calc_psnrs(all_vols, ax_cr_sg)  
    cmap = mpl.cm.get_cmap('jet_r')
    (min_psnr, max_psnr) = (psnrs.min(), min(psnrs.max(),100))
    norm = mpl.colors.Normalize(vmin=min_psnr, vmax=max_psnr)
    nrows = 2 if plot_most_fluc else 4
    ncols = int(2*nrows) if ax_cr_sg == 0 else int(2.5*nrows)
    figsize = (16,6.5) if (ax_cr_sg == 0 and plot_most_fluc) else (16,7.5) if (ax_cr_sg == 0 and (not plot_most_fluc))\
        else (16,8.66) if (ax_cr_sg != 0 and plot_most_fluc) else (16,12)
    
    filenames = []
    for t in np.arange(0, 50, 18):#all_vols.shape[0]):
        filename = f'{ind_ims_dir}/frame_{t}.png'
        filenames.append(filename)
        
        # last frame of each viz stays longer
        if (t == (all_vols.shape[0]-1)):
            for t in range(5):
                filenames.append(filename)
            
        # save img
        sl_nos = psnrs.min(0).argsort()[:nrows*ncols] if plot_most_fluc else np.arange(0,(nrows*ncols-1)*(psnrs.shape[1]//(nrows*ncols-1))+1,psnrs.shape[1]//(nrows*ncols-1))
        fig,ax = plt.subplots(nrows,ncols, figsize=figsize)
        for i in range(nrows):
            for j in range(ncols):
                sl_no = sl_nos[ncols*i+j]
                if ax_cr_sg == 0:
                    im_to_show = all_vols[t,...,sl_no]
                elif ax_cr_sg == 1:
                    im_to_show = all_vols[t,:,sl_no,:]
                elif ax_cr_sg == 2:
                    im_to_show = all_vols[t,sl_no,:,:]
                im = ax[i,j].imshow(im_to_show,cmap='gray', interpolation='none')
                ax[i,j].axis('off')
                ps = psnrs[t,sl_no]
                ps_color = cmap((ps-min_psnr)/(max_psnr-min_psnr))
                # Create a Rectangle patch
                rect = patches.Rectangle((0, 0), im_to_show.shape[1], im_to_show.shape[0], linewidth=5, edgecolor=ps_color, facecolor='none')
                # Add the patch to the Axes
                ax[i,j].add_patch(rect)
                #print(f'max is {min_psnr}, min is {max_psnr}')
                ax[i,j].set_title('Slice {}'.format(sl_no))
                ax[i,j].text(0.5,-0.1, '({:.1f} dB)'.format(ps), color=ps_color, size=10, ha="center", transform=ax[i,j].transAxes)
                divider = make_axes_locatable(ax[i,j])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im, cax=cax, orientation='vertical')
        plt.subplots_adjust(left=0.01, right=0.90, bottom=0.05, top=0.935, wspace=0.32)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
        cb1 = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='vertical')
        cb1.set_label('PSNR wrt the Initial Image (dB)')
        plt.show()
        if plot_most_fluc:
            if ax_cr_sg == 0:
                plt.suptitle(f"Most Fluctuating Axial Images ({pt_id}, Time Point: {t})")
            elif ax_cr_sg == 1:
                plt.suptitle(f"Most Fluctuating Coronal Images ({pt_id}, Time Point: {t})")
            elif ax_cr_sg == 2:
                plt.suptitle(f"Most Fluctuating Sagittal Images ({pt_id}, Time Point: {t})")
        else:
            if ax_cr_sg == 0:
                plt.suptitle(f"Axial Images ({pt_id}, Time Point: {t})")
            elif ax_cr_sg == 1:
                plt.suptitle(f"Coronal Images ({pt_id}, Time Point: {t})")
            elif ax_cr_sg == 2:
                plt.suptitle(f"Sagittal Images ({pt_id}, Time Point: {t})")        
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
    # print('Removing Images\n')
    # Remove files
    # for filename in set(filenames):
    #     os.remove(filename)
    print('DONE')
    
    
if __name__ == "__main__":
    main()