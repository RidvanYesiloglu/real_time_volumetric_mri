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
import time

def calc_psnrs_wrti(all_vols, ax_cr_sg):
    if ax_cr_sg == 0:
        return 20*np.log((all_vols[0:1]**2).max((1,2)))-10*np.log(((all_vols[0:1]-all_vols)**2).mean((1,2)))
    elif ax_cr_sg == 1:
        return 20*np.log((all_vols[0:1]**2).max((1,3)))-10*np.log(((all_vols[0:1]-all_vols)**2).mean((1,3)))
    elif ax_cr_sg == 2:
        return 20*np.log((all_vols[0:1]**2).max((2,3)))-10*np.log(((all_vols[0:1]-all_vols)**2).mean((2,3)))
    
def main(all_vols, pt_id, ax_cr_sg, plot_most_fluc=False):
    start_time = time.perf_counter()
    im_type_str = 'axial' if ax_cr_sg == 0 else 'coronal' if ax_cr_sg == 1 else 'sagittal' if ax_cr_sg == 2 else 'ERROR'
    gif_name = f'most_fluc_{im_type_str}s_vs_t_{pt_id}' if plot_most_fluc else f'all_{im_type_str}s_vs_t_{pt_id}'
    print(f'The gif {gif_name} is being created.')
    gifs_dir = f'/raid/yesiloglu/data/real_time_volumetric_mri/{pt_id}/temporal_evol_gifs'
    ind_ims_dir = f'{gifs_dir}/{gif_name}_ims'
    if not os.path.exists(ind_ims_dir):
        print(f'Directory created: {ind_ims_dir}')
        os.makedirs(ind_ims_dir)
    else:
        print(f'Directory already exists, will be overwritten: {ind_ims_dir}')
    # Calculate PSNRs:
    print(f'{im_type_str.capitalize()} PSNRs are being calculated.')
    psnrs = calc_psnrs_wrti(all_vols, ax_cr_sg)
    print(psnrs[547,30:40])
    curr_time = time.perf_counter()
    print(f"Elapsed total for the gif {gif_name}: {curr_time-start_time} seconds")
    if not plot_most_fluc:
        # Create and save the plot of PSNRs:
        print(f'Creating and saving the plot of {im_type_str.capitalize()} PSNRs.')
        fig,ax = plt.subplots(figsize=(9,8))
        sl_no_mplier = 5
        im = ax.imshow(np.repeat(psnrs, sl_no_mplier, axis=1))
        ax.set_title(f'{im_type_str.capitalize()} PSNRs wrt the Initial Image')
        ax.set_xlabel('Slice No')
        ax.set_xticks(sl_no_mplier*np.arange(0,psnrs.shape[1],step=10) + sl_no_mplier//2)
        ax.set_xticklabels(np.arange(0,psnrs.shape[1],step=10))
        ax.set_ylabel('Time Index')
        ax.set_yticks(np.arange(0,psnrs.shape[0],step=50))
        ax.set_yticklabels(np.arange(0,psnrs.shape[0],step=50))
        fig.colorbar(im, ax=ax, orientation='vertical')
        plt.show()
        plt.savefig(f'{gifs_dir}/{im_type_str}_psnrs_wrt_init.pdf', dpi=60, bbox_inches='tight')
        plt.close()
    
    # # Create gif
    # print(f'Creating and saving the gif {gif_name}.')
    # cmap = mpl.cm.get_cmap('jet_r')
    # (min_psnr, max_psnr) = (psnrs.min(), min(psnrs.max(),100))
    # norm = mpl.colors.Normalize(vmin=min_psnr, vmax=max_psnr)
    # nrows = 2 if plot_most_fluc else 4
    # ncols = int(2*nrows) if ax_cr_sg == 0 else int(2.5*nrows)
    # figsize = (16,7.5) if (ax_cr_sg == 0 and plot_most_fluc) else (16,7.5) if (ax_cr_sg == 0 and (not plot_most_fluc))\
    #     else (16,9.5) if (ax_cr_sg != 0 and plot_most_fluc) else (19.5,11)
    # filenames = []
    # for t in np.arange(0, all_vols.shape[0]):
    #     filename = f'{ind_ims_dir}/frame_{t}.png'
    #     filenames.append(filename)
        
    #     # last frame stays longer
    #     if (t == (all_vols.shape[0]-1)):
    #         for t in range(5):
    #             filenames.append(filename)
            
    #     # create the image for time=t
    #     sl_nos = psnrs.min(0).argsort()[:nrows*ncols] if plot_most_fluc else np.arange(0,(nrows*ncols-1)*(psnrs.shape[1]//(nrows*ncols-1))+1,psnrs.shape[1]//(nrows*ncols-1))
    #     fig,ax = plt.subplots(nrows,ncols, figsize=figsize)
    #     for i in range(nrows):
    #         for j in range(ncols):
    #             sl_no = sl_nos[ncols*i+j]
    #             if ax_cr_sg == 0:
    #                 im_to_show = all_vols[t,...,sl_no]
    #             elif ax_cr_sg == 1:
    #                 im_to_show = all_vols[t,:,sl_no,:]
    #             elif ax_cr_sg == 2:
    #                 im_to_show = all_vols[t,sl_no,:,:]
    #             im = ax[i,j].imshow(im_to_show,cmap='gray', interpolation='none')
    #             ax[i,j].axis('off')
    #             ps = psnrs[t,sl_no]
    #             ps_color = cmap((ps-min_psnr)/(max_psnr-min_psnr))
    #             # Create a rectangle patch around the image to indicate the PSNR wrt the initial image
    #             rect = patches.Rectangle((0, 0), im_to_show.shape[1], im_to_show.shape[0], linewidth=5, edgecolor=ps_color, facecolor='none')
    #             ax[i,j].add_patch(rect)
    #             ax[i,j].set_title('Slice {}'.format(sl_no))
    #             ax[i,j].text(0.5,-0.1+0.05*plot_most_fluc-0.01*(ax_cr_sg==0)+0.03*((ax_cr_sg!=0)and(not plot_most_fluc)), '({:.1f} dB)'.format(ps), color=ps_color, size=10, ha="center", transform=ax[i,j].transAxes)
    #             divider = make_axes_locatable(ax[i,j])
    #             cax = divider.append_axes('right', size='5%', pad=0.05)
    #             fig.colorbar(im, cax=cax, orientation='vertical')
    #     plt.subplots_adjust(left=0.01, right=0.90, bottom=0.05, top=0.935, wspace=0.32)
    #     cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    #     cb1 = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='vertical')
    #     cb1.set_label('PSNR wrt the Initial Image (dB)')
    #     if plot_most_fluc:
    #         plt.suptitle(f"Most Fluctuating {im_type_str.capitalize()} Images ({pt_id}, Time Point: {t:3d})")
    #     else:
    #         plt.suptitle(f"{im_type_str.capitalize()} Images ({pt_id}, Time Point: {t:3d})")      
    #     plt.show()
    #     plt.savefig(filename, dpi=96, bbox_inches='tight')
    #     plt.close()

    # with imageio.get_writer(f'{gifs_dir}/{gif_name}.gif', mode='I') as writer:
    #     for filename in filenames:
    #         image = imageio.imread(filename)
    #         writer.append_data(image)
    curr_time = time.perf_counter()
    print(f"Elapsed total for the gif {gif_name}: {curr_time-start_time} seconds, done.")
    
if __name__ == "__main__":
    main()