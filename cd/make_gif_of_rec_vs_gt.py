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
def mean_squared_err_2d(reconstructed, original, ax_cr_sg): # input, target
    if ax_cr_sg == 0:
        return ((reconstructed-original)**2).mean((0,1))
    elif ax_cr_sg == 1:
        return ((reconstructed-original)**2).mean((0,2))
    elif ax_cr_sg == 2:
        return ((reconstructed-original)**2).mean((1,2))
def calc_psnrs_2d(reconstructed, original, ax_cr_sg):
    if ax_cr_sg == 0:
        return 20*np.log((original**2).max((0,1)))-10*np.log(((original-reconstructed)**2).mean((0,1)))
    elif ax_cr_sg == 1:
        return 20*np.log((original**2).max((0,2)))-10*np.log(((original-reconstructed)**2).mean((0,2)))
    elif ax_cr_sg == 2:
        return 20*np.log((original**2).max((1,2)))-10*np.log(((original-reconstructed)**2).mean((1,2)))
def main(output_im, ref_im, step, ax_cr_sg, pt_id, res_dir, args, repr_str, plot_max_mse=False):
    start_time = time.perf_counter()
    im_type_str = 'axial' if ax_cr_sg == 0 else 'coronal' if ax_cr_sg == 1 else 'sagittal' if ax_cr_sg == 2 else 'ERROR'
    gif_name = f'max_mse_{im_type_str}s_vs_t_{pt_id}' if plot_max_mse else f'all_{im_type_str}s_vs_t_{pt_id}'
    print(f'The gif {gif_name} is being created.')


    # Calculate PSNRs:
    print(f'{im_type_str.capitalize()} MSEs are being calculated.')
    mses = mean_squared_err_2d(output_im, ref_im, ax_cr_sg)

    curr_time = time.perf_counter()
    print(f"Elapsed total for the gif {gif_name}: {curr_time-start_time} seconds")
    
    # Create gif
    print(f'Creating and saving the gif {gif_name}.')

    nrows = 2 if plot_max_mse else 4
    ncols = int(2*nrows) if ax_cr_sg == 0 else int(2.5*nrows)
    sl_nos = mses.argsort()[-nrows*ncols:] if plot_max_mse else np.arange(0,(nrows*ncols-1)*(mses.shape[1]//(nrows*ncols-1))+1,mses.shape[0]//(nrows*ncols-1))
    psnrs = calc_psnrs_2d(output_im, ref_im, ax_cr_sg)
    cmap = mpl.cm.get_cmap('jet_r')
    (min_psnr, max_psnr) = (psnrs.min(), min(psnrs.max(),100))
    norm = mpl.colors.Normalize(vmin=min_psnr, vmax=max_psnr)
    figsize = (16,7.5) if (ax_cr_sg == 0 and plot_max_mse) else (16,7.5) if (ax_cr_sg == 0 and (not plot_max_mse))\
        else (16,9.5) if (ax_cr_sg != 0 and plot_max_mse) else (19.5,11)

    fig,ax = plt.subplots(nrows,ncols, figsize=figsize)
    for i in range(nrows):
        for j in range(ncols):
            sl_no = sl_nos[ncols*i+j]
            if ax_cr_sg == 0:
                im_to_show = np.concatenate((output_im[...,sl_no],ref_im[...,sl_no]),1)
            elif ax_cr_sg == 1:
                im_to_show = np.concatenate((output_im[:,sl_no,:],ref_im[:,sl_no,:]),1)
            elif ax_cr_sg == 2:
                im_to_show = np.concatenate((output_im[sl_no,:,:],ref_im[sl_no,:,:]),1)
            im = ax[i,j].imshow(im_to_show, cmap='gray', interpolation='none', vmin=im_to_show.min(), vmax=im_to_show.max())
            ax[i,j].axis('off')
            ps = psnrs[sl_no]
            ps_color = cmap((ps-min_psnr)/(max_psnr-min_psnr))
            # Create a rectangle patch around the image to indicate the PSNR wrt the initial image
            rect = patches.Rectangle((0, 0), im_to_show.shape[1], im_to_show.shape[0], linewidth=5, edgecolor=ps_color, facecolor='none')
            ax[i,j].add_patch(rect)
            ax[i,j].set_title('Slice {}'.format(sl_no))
            ax[i,j].text(0.5,-0.1+0.05*plot_max_mse-0.01*(ax_cr_sg==0)+0.03*((ax_cr_sg!=0)and(not plot_max_mse)), '({:.1f} dB)'.format(ps), color=ps_color, size=10, ha="center", transform=ax[i,j].transAxes)
            divider = make_axes_locatable(ax[i,j])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
    plt.subplots_adjust(left=0.01, right=0.90, bottom=0.05, top=0.935, wspace=0.32)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cb1 = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='vertical')
    cb1.set_label('PSNR wrt the Initial Image (dB)')
    if plot_max_mse:
        plt.suptitle(f"Most Fluctuating {im_type_str.capitalize()} Images ({pt_id}, Time Point: {args.im_ind:3d}, Epoch: {ep_no:3d})")
    else:
        plt.suptitle(f"{im_type_str.capitalize()} Images ({pt_id}, Time Point: {args.im_ind:3d})")      
    plt.show()
    plt.savefig(f'{res_dir}/{gif_name}', dpi=96, bbox_inches='tight')
    plt.close()
    # with imageio.get_writer(f'{res_dir}{gif_name}.gif', mode='I') as writer:
    #     for filename in filenames:
    #         image = imageio.imread(filename)
    #         writer.append_data(image)
    curr_time = time.perf_counter()
    print(f"Elapsed total for the gif {gif_name}: {curr_time-start_time} seconds, done.")
    
if __name__ == "__main__":
    main()