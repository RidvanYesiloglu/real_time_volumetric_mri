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
from skimage.metrics import structural_similarity as ssim

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
def calc_ssim(rec, ref):
    test_ssim = ssim(rec, ref, data_range=ref.max()-ref.min())
    return test_ssim
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
    ssims = np.zeros((len(sps)*len(ts), t_end-t_st+1))
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
                ssims[conf_ind,time_ind - t_st] = calc_ssim(recs[conf_ind,time_ind - t_st], refs[time_ind - t_st])
            conf_ind += 1
            
    return recs, refs, psnrs, ssims  
# TO DO: draw border around subplots, draw a main, big horizontal and vertical axis
def make_gif_frames(args, recs, refs, psnrs, ssims, sps, ts, ax_cr_sg, sl_no, gif_dir, gif_name, t_st):
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
        (min_ssim, max_ssim) = (ssims[:,t].min(), ssims[:,t].max())
        bps_conf_no = psnrs[:,t].argmax()
        bss_conf_no = ssims[:,t].argmax()
        norm = mpl.colors.Normalize(vmin=min_psnr, vmax=max_psnr)
        fig,ax = plt.subplots(nrows,ncols, figsize=figsize)
        # ims_ax = fig.add_axes([0, 0, 0.55, 1])
        # ims_ax.set_xlabel('Time Continuity Coefficient')
        # ims_ax.set_ylabel('Spatial Continuity Coefficient')
        # ims_ax.set_xticks(np.arange(len(ts)))
        # ims_ax.set_xticklabels([str(tcc) for tcc in ts])
        # ims_ax.set_yticks(np.arange(len(sps)))
        # ims_ax.set_yticklabels([str(sp) for sp in sps])
        for i in range(nrows):
            for j in range(ncols):
                conf_no = ncols*i+j
                im_to_show = np.concatenate((recs[conf_no, t], refs[t]),1)
                im_to_show[im_to_show<0]=0
                im = ax[i,j].imshow(im_to_show,cmap='gray', interpolation='none')#, vmin=immin, vmax=immax)
                ax[i,j].axis('off')
                ps = psnrs[conf_no,t]
                ss = ssims[conf_no,t]
                ps_color = cmap((ps-min_psnr)/(max_psnr-min_psnr))
                # Create a rectangle patch around the image to indicate the PSNR wrt the initial image
                # rect = patches.Rectangle((0,0), im_to_show.shape[1], im_to_show.shape[0], linewidth=5, edgecolor=ps_color, facecolor='none')
                # ax[i,j].add_patch(rect)
                
                # Top patch
                rect_top = patches.Rectangle(
                    (-5,-15),
                    138,
                    153,
                    transform=ax[i,j].transData,
                    color=ps_color,
                    linewidth=1,
                    fill=True,
                    zorder=-100,
                )
                fig.patches.append(rect_top)
                # rect_top = patches.Rectangle((0, -5), im_to_show.shape[1], 5, linewidth=5, edgecolor=ps_color, facecolor=ps_color, transform=ax[i,j].transData, zorder=2)
                # ax[i,j].add_patch(rect_top)
                # Bottom patch
                # rect_bot = patches.ConnectionPatch(
                #     (10,138),
                #     (118,138),
                #     coordsA=ax[i,j].transData,
                #     coordsB=ax[i,j].transData,
                #     color=ps_color,
                #     linewidth=20,
                #     zorder=-100,
                # )
                # fig.patches.append(rect_bot)
                # rect_bot = patches.Rectangle((0, 128), im_to_show.shape[1], 2, linewidth=5, edgecolor=ps_color, facecolor=ps_color)
                # ax[i,j].add_patch(rect_bot)
                if conf_no == bss_conf_no:
                    ax[i,j].set_title(f'TCC: {ts[j]:.0e}, SCC: {sps[i]:.0e}', color='black')
                else:
                    ax[i,j].set_title(f'TCC: {ts[j]:.0e}, SCC: {sps[i]:.0e}', color='black')
                if i == (nrows-1): #-0.29
                    ax[i,j].text(64, 155, f'{ts[j]:.0e}', color='red', size=14, ha="center", va="center", transform=ax[i,j].transData)
                    if j == ((ncols-1)//2):
                        ax[i,j].text(140,170, 'Time Continuity Reg. Coefficient', color='red', size=18, ha="center", va="center", transform=ax[i,j].transData)
                if j == 0:#-22
                    ax[i,j].text(-40,64, f'{sps[i]:.0e}', color='red', size=14, ha="center", va="center", transform=ax[i,j].transData)
                    if i == ((nrows-1)//2):
                        ax[i,j].text(-65,140, 'Spatial Continuity Reg. Coefficient', color='red', size=18, ha="center", va="center", rotation='vertical', transform=ax[i,j].transData)
                ax[i,j].text(0.5,-0.1-0.01*(ax_cr_sg==0)+0.03*(ax_cr_sg!=0), '({:.1f} dB, {:.3f})'.format(ps, ss), color='w', size=11, ha="center", transform=ax[i,j].transAxes)
                # divider = make_axes_locatable(ax[i,j])
                # cax = divider.append_axes('right', size='5%', pad=0.05)
                # fig.colorbar(im, cax=cax, orientation='vertical')
        plt.subplots_adjust(left=0.04, right=0.55, bottom=0.05, top=0.935, wspace=0.09, hspace=0.14)
        tcc_arrow = patches.ConnectionPatch(
            (0,145),
            (128,145),
            coordsA=ax[-1,0].transData,
            coordsB=ax[-1,-1].transData,
            # Default shrink parameter is 0 so can be omitted
            color="red",
            arrowstyle="-|>",  # "normal" arrow
            mutation_scale=30,  # controls arrow head size
            linewidth=3,
        )
        fig.patches.append(tcc_arrow)
        for i in range(ncols):
            tcc_tick = patches.ConnectionPatch(
                (64,143),
                (64,147),
                coordsA=ax[-1,i].transData,
                coordsB=ax[-1,i].transData,
                color="red",
                linewidth=3,
            )
            fig.patches.append(tcc_tick)
        scc_arrow = patches.ConnectionPatch(
            (-15,0),
            (-15,128),
            coordsA=ax[0,0].transData,
            coordsB=ax[-1,0].transData,
            # Default shrink parameter is 0 so can be omitted
            color="red",
            arrowstyle="-|>",  # "normal" arrow
            mutation_scale=30,  # controls arrow head size
            linewidth=3,
        )
        fig.patches.append(scc_arrow)
        for i in range(nrows):
            scc_tick = patches.ConnectionPatch(
                (-18,64),
                (-12,64),
                coordsA=ax[i,0].transData,
                coordsB=ax[i,0].transData,
                color="red",
                linewidth=3,
            )
            fig.patches.append(scc_tick)
            
        cbar_ax = fig.add_axes([0.55, 0.15, 0.02, 0.7])
        cb1 = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='vertical')
        cb1.set_label('PSNR (dB)')
        # left bottom graph: psnr vs sp
        graph_ax = fig.add_axes([0.63, 0.25, 0.15, 0.2])
        for no,tcc in enumerate(ts):
            graph_ax.plot(psnrs[no::ncols,t], label=f'TCC:{tcc}')
        graph_ax.set_title('PSNR vs SCC (Spatial Continuity Coefficient)')
        graph_ax.set_ylabel(f'PSNR at t={t}')
        #
        graph_ax = fig.add_axes([0.63, 0.05, 0.15, 0.2])
        for no,tcc in enumerate(ts):
            graph_ax.plot(psnrs[no::ncols,:].mean(1), label=f'TCC:{tcc}')
        graph_ax.set_ylabel('Av. PSNR')
        graph_ax.set_xlabel('SCC')
        graph_ax.set_xticks(np.arange(len(sps)))
        graph_ax.set_xticklabels([str(sp) for sp in sps])
        graph_ax.legend()
        # right bottom graph: ssim vs sp
        graph_ax = fig.add_axes([0.85, 0.25, 0.15, 0.2])
        for no,tcc in enumerate(ts):
            graph_ax.plot(ssims[no::ncols,t], label=f'TCC:{tcc}')
        graph_ax.set_title('SSIM vs SCC (Spatial Continuity Coefficient)')
        graph_ax.set_ylabel(f'SSIM at t={t}')
        # 
        graph_ax = fig.add_axes([0.85, 0.05, 0.15, 0.2])
        for no,tcc in enumerate(ts):
            graph_ax.plot(ssims[no::ncols,:].mean(1), label=f'TCC:{tcc}')
        graph_ax.set_ylabel('Av. SSIM')
        graph_ax.set_xlabel('SCC')
        graph_ax.set_xticks(np.arange(len(sps)))
        graph_ax.set_xticklabels([str(sp) for sp in sps])
        graph_ax.legend()
        # left top graph: psnr vs tcc
        graph_ax = fig.add_axes([0.63, 0.72, 0.15, 0.2])
        for no,scc in enumerate(sps):
            graph_ax.plot(psnrs[no*ncols:(no+1)*ncols,t], label=f'SCC:{scc}')
        graph_ax.set_title('PSNR vs TCC (Time Continuity Coefficient)')
        graph_ax.set_ylabel(f'PSNR at t={t}')
        # 
        graph_ax = fig.add_axes([0.63, 0.52, 0.15, 0.2])
        for no,scc in enumerate(sps):
            graph_ax.plot(psnrs[no*ncols:(no+1)*ncols,:].mean(1), label=f'SCC:{scc}')
        graph_ax.set_ylabel('Av. PSNR')
        graph_ax.set_xlabel('TCC')
        graph_ax.set_xticks(np.arange(len(ts)))
        graph_ax.set_xticklabels([str(tcc) for tcc in ts])
        graph_ax.legend()
        # right bottom graph: ssim vs tcc
        graph_ax = fig.add_axes([0.85, 0.72, 0.15, 0.2])
        for no,scc in enumerate(sps):
            graph_ax.plot(ssims[no*ncols:(no+1)*ncols,t], label=f'SCC:{scc}')
        graph_ax.set_title('SSIM vs TCC (Time Continuity Coefficient)')
        graph_ax.set_ylabel(f'SSIM at t={t}')
        # 
        graph_ax = fig.add_axes([0.85, 0.52, 0.15, 0.2])
        for no,scc in enumerate(sps):
            graph_ax.plot(ssims[no*ncols:(no+1)*ncols,:].mean(1), label=f'SCC:{scc}')
        graph_ax.set_ylabel('Av. SSIM')
        graph_ax.set_xlabel('TCC')
        graph_ax.set_xticks(np.arange(len(ts)))
        graph_ax.set_xticklabels([str(tcc) for tcc in ts])
        graph_ax.legend()
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
    ax_cr_sg = int(input('Axial (0), coronal (1), sagittal (2)? '))
    sl_no = int(input('Slice no: '))
    t_st = args.st_ind
    t_end = args.end_ind
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
            recs, refs, psnrs, ssims = find_recs_for_sps_ts(args, params_dict, sps, ts, ax_cr_sg, sl_no, t_st, t_end) #(16,12,128,64) or (16,12,128,128) and  psnrs: (16,12)
            gif_dir = f'{args.main_folder}{args.pt}/reg_gifs/sps_ts_wt{wt}_jc{jc}/'
            gif_name = f'sl{sl_no}_sps_ts_wt{wt}_jc{jc}'
            print(f'Reconstructions found. Making the gif: {gif_name}')
            make_gif_frames(args, recs, refs, psnrs, ssims, sps, ts, ax_cr_sg, sl_no, gif_dir, gif_name, t_st)
            print('Gif made.')

if __name__ == "__main__":
    main() 
    
    