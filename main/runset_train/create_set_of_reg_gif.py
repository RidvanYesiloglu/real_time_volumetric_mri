from runset_train import train
import torch
import pickle
import os
import runset_train.parameters as parameters
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import imageio
import glob
from argparse import Namespace
def get_parameters_of_runs(params_dict):
    # First create cart prod runsets.
    vals_list = []
    cart_prod_runsets = []
    for info in params_dict.param_infos:
        cart_prod_runsets, vals_list = info.get_input_and_update_runsets(cart_prod_runsets, vals_list, params_dict)
    # Then create args list.
    args_list = []
    indRunNo = 1
    for run in cart_prod_runsets:
        kwargs = {}
        for no, name in enumerate([info.name for info in params_dict.param_infos]): kwargs[name] = run[no]
        kwargs['indRunNo'] = indRunNo # add individual run no
        kwargs['totalInds'] = len(cart_prod_runsets)
        curr_args = Namespace(**kwargs)
        args_list.append(curr_args)
        indRunNo += 1
    return args_list

def find_total_runs(wts, sps, jcs, ts):
    curr_ind = 0
    for wt in wts:
        for sp in sps:
            for jc in jcs:
                for t in ts:
                    if wt==0 and jc != 0:
                        continue
                    curr_ind += 1

                    print(wt, sp, jc, (jc!=0))
                    
    return curr_ind
def calc_psnr(rec, ref):
    test_psnr = 20*np.log10(ref.max()) - 10 * np.log10(((rec-ref)**2).mean())
    return test_psnr
def find_recs_for_sps_ts(args, params_dict, sps, ts, ax_cr_sg, sl_no, t_st, t_end):
    pt_dir = f'{args.main_folder}{args.pt}/'
    im_dim = (128,128) if ax_cr_sg==0 else (128,64)
    recs = np.zeros((len(sps)*len(ts), t_end-t_st+1, im_dim[0], im_dim[1]))
    ref_dir = args.data_dir + args.pt + '/all_vols.npy'
    if ax_cr_sg == 0:
        refs = np.load(ref_dir)[t_st:t_end+1].astype('float32')[:,:,:,sl_no]
    elif ax_cr_sg == 1:
        refs = np.load(ref_dir)[t_st:t_end+1].astype('float32')[:,:,sl_no,:]
    elif ax_cr_sg == 2:
        refs = np.load(ref_dir)[t_st:t_end+1].astype('float32')[:,sl_no,:,:]
    psnrs = np.zeros((len(sps)*len(ts), t_end-t_st+1))
    conf_ind = 0
    for sp in sps:
        args.use_sp_cont_reg = (sp!=0)
        args.lambda_sp = sp
        for t in ts:
            args.use_t_cont_reg = (t!=0)    
            args.lambda_t = t
            for time_ind in range(t_st, t_end+1):
                args.im_ind = time_ind
                repr_str = parameters.create_repr_str(args, [info.name for info in params_dict.param_infos], wantShort=True, params_dict=params_dict)
                res_dir = f'{pt_dir}{args.conf}/t_{args.im_ind}/{repr_str}'
                try:
                    loaded_rec = np.load(glob.glob(os.path.join(res_dir, 'rec_*'))[0]).squeeze()
                    # print('LOADED! is_zeros: ', ((loaded_rec**2).sum()==0))
                    # inpp = (input('Bekliyorum'))
                except Exception as e:
                    # print('Error in loading:', e)
                    # print('Res dir:', res_dir)
                    # print('len: ', len(glob.glob(os.path.join(res_dir, 'rec_*'))))
                    # try:
                    #     print('ilk eleman', glob.glob(os.path.join(res_dir, 'rec_*'))[0])
                    # except Exception as e2:
                    #     print('ilk yok, error:', e2)
                    # inpp = (input('Bekliyorum'))
                    loaded_rec = np.zeros((128,128,64))
                if ax_cr_sg == 0:
                    recs[conf_ind,time_ind - t_st] = loaded_rec[:,:,sl_no]
                elif ax_cr_sg == 1:
                    recs[conf_ind,time_ind - t_st] = loaded_rec[:,sl_no,:]
                elif ax_cr_sg == 2:
                    recs[conf_ind,time_ind - t_st] = loaded_rec[sl_no,:,:]
                psnrs[conf_ind,time_ind - t_st] = calc_psnr(recs[conf_ind,time_ind - t_st], refs[time_ind - t_st])
                
            conf_ind += 1
            
    return recs, refs, psnrs   
# TO DO: draw border around subplots, draw a main, big horizontal and vertical axis
def make_gif_frames(args, recs, refs, psnrs, sps, ts, ax_cr_sg, sl_no, gif_dir, gif_name, t_st):
    filenames = []
    cmap = mpl.cm.get_cmap('jet')
    
    nrows = 4
    ncols = 4
    figsize = (16,7.5) if (ax_cr_sg == 0) else (19.5,11)
    im_type_str = 'axial' if ax_cr_sg == 0 else 'coronal' if ax_cr_sg == 1 else 'sagittal' if ax_cr_sg == 2 else 'ERROR'
    ind_ims_dir = f'{gif_dir}{gif_name}_ims/'
    if not os.path.exists(ind_ims_dir):
        os.makedirs(ind_ims_dir)
    #print('recs:', recs.shape, 'refs:', refs.shape) #(16,12,128,64) , (12,128,64)
    for t in np.arange(0, recs.shape[1]):
        filename = f'{ind_ims_dir}/frame_{t}.png'
        filenames.append(filename)
        (min_psnr, max_psnr) = (psnrs[:,t].min(), min(psnrs[:,t].max(),100))
        best_conf_no = psnrs[:,t].argmax()
        norm = mpl.colors.Normalize(vmin=min_psnr, vmax=max_psnr)
        fig,ax = plt.subplots(nrows,ncols, figsize=figsize)
        for i in range(nrows):
            for j in range(ncols):
                conf_no = ncols*i+j
                im_to_show = np.concatenate((recs[conf_no, t], refs[t]),1)
                im_to_show[im_to_show<0]=0
                im = ax[i,j].imshow(im_to_show,cmap='gray', interpolation='none')#, vmin=immin, vmax=immax)
                ax[i,j].axis('off')
                ps = psnrs[conf_no,t]
                ps_color = cmap((ps-min_psnr)/(max_psnr-min_psnr))
                # Create a rectangle patch around the image to indicate the PSNR wrt the initial image
                rect = patches.Rectangle((0, 0), im_to_show.shape[1], im_to_show.shape[0], linewidth=5, edgecolor=ps_color, facecolor='none')
                ax[i,j].add_patch(rect)
                if conf_no == best_conf_no:
                    ax[i,j].set_title(f'Time CC: {ts[j]}, Spat CC: {sps[i]} (BEST)', color='r')
                else:
                    ax[i,j].set_title(f'Time CC: {ts[j]}, Spat CC: {sps[i]}')
                ax[i,j].text(0.5,-0.1-0.01*(ax_cr_sg==0)+0.03*(ax_cr_sg!=0), '({:.1f} dB)'.format(ps), color=ps_color, size=10, ha="center", transform=ax[i,j].transAxes)
                divider = make_axes_locatable(ax[i,j])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im, cax=cax, orientation='vertical')
        plt.subplots_adjust(left=0.01, right=0.90, bottom=0.05, top=0.935, wspace=0.32)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
        cb1 = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='vertical')
        cb1.set_label('PSNR (dB)')
        if args.conf == 'trn_w_trns':
            plt.suptitle(f"{im_type_str.capitalize()} Images vs Spatial and Temporal Continuity Loss Coefficients ({args.pt} - With Transformation NeRP - JC Loss Coef. on Grid: {args.lambda_JR} - Time Point: {(t+t_st):3d})")
        else:
            plt.suptitle(f"{im_type_str.capitalize()} Images vs Spatial and Temporal Continuity Loss Coefficients ({args.pt} - Without Transformation NeRP - Time Point: {(t+t_st):3d})")
        plt.show()
        plt.savefig(filename, dpi=96, bbox_inches='tight')
        plt.close()
    with imageio.get_writer(f'{gif_dir}/{gif_name}.gif', mode='I', duration=1.09) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

def main():
    params_dict = parameters.decode_arguments_dictionary('params_dictionary')
    args_list = get_parameters_of_runs(params_dict)
    args = args_list[0]
    if args.end_ind == -1:
        args.end_ind = np.load(args.data_dir+args.pt+'/all_vols.npy').shape[0] - 1
        print(f'Ending index was made: {args.end_ind} (which is the last data point over time.)')
    ax_cr_sg = int(input('Axial (0), coronal (1), sagittal (2)?'))
    sl_no = int(input('Slice no:'))
    t_st = 1
    t_end = 12
    wts = [0,1]
    sps = [0,1e2,1e3,1e4]
    jcs = [0,1e2,1e3,1e4]
    ts = [0,1e2,1e3,1e4] 
    for wt in wts:
        for jc in jcs:
            if wt==0 and jc != 0:
                continue
            args.conf = 'trn_wo_trns' if wt==0 else 'trn_w_trns'
            args.use_jc_grid_reg = (jc!=0)
            args.lambda_JR = jc
            print(f'Current wt:{wt}, jc: {jc}. Finding reconstructions...')
            recs, refs, psnrs = find_recs_for_sps_ts(args, params_dict, sps, ts, ax_cr_sg, sl_no, t_st, t_end) #(16,12,128,64) or (16,12,128,128) and  psnrs: (16,12)
            gif_dir = f'{args.main_folder}{args.pt}/reg_gifs/sl_{sl_no}/'
            gif_name = f'sps_ts_wt{wt}_jc{jc}'
            print(f'Reconstructions found. Making the gif: {gif_name}')
            make_gif_frames(args, recs, refs, psnrs, sps, ts, ax_cr_sg, sl_no, gif_dir, gif_name, t_st)
            print('Gif made.')

if __name__ == "__main__":
    main() 
    
    