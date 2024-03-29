import os
import torch
import numpy as np

import time
import multiprocessing
from models.nerp.model_nerp import Main_Module as Main_Module
from models.nerp.plot_nerp import plot_change_of_value
import make_gif_of_rec_vs_gt

from utils import conv_repr_str_to_mlt_line, PSNR, check_gpu

from torchnufftexample import create_radial_mask, project_radial, backproject_radial
import sys
import torch.nn as nn

import glob
# inps_dict: {'save_folder': save_folder, 'args':args, 'repr_str':repr_str, 'run_number':run_number, 'model': model}
# outs_dict: 
def prerun_i_actions(inps_dict):
    args =inps_dict['args']
    main_module = Main_Module(args)
    main_module.eval()
    test_psnr, test_ssim, test_loss = main_module.test_psnr_ssim()
    main_module.train()
    
    init_psnr_str = 'Initial loss: {:.4f}, initial psnr: {:.4f}, initial ssim: {:.4f}\n'.format(test_loss, test_psnr, test_ssim)
    print(init_psnr_str)
    mlt_line_long_repr = ''.join(list(conv_repr_str_to_mlt_line(inps_dict['long_repr_str'], '&')))

    r_logs = open(os.path.join(inps_dict['res_dir'], 'logs_r{}.txt'.format(inps_dict['run_number'])), "a")
    r_logs.write('Runset Name: {}\n'.format(args.rsName))
    r_logs.write('Short representation string for parameters set:{}\n'.format(inps_dict['repr_str']))
    r_logs.write('All parameters:\n{}'.format(mlt_line_long_repr))
    r_logs.write(init_psnr_str)
    r_logs.close()
    
    preruni_dict={'main_module':main_module}

    #np.save(os.path.join(inps_dict['save_folder'], 'pretrainmodel_out'), test_output.detach().cpu().numpy())
    return preruni_dict

def print_freq_actions(inps_dict, preruni_dict):
    args =inps_dict['args']
    preruni_dict['main_module'].eval()
    test_psnr, test_ssim, test_loss = preruni_dict['main_module'].test_psnr_ssim()
    preruni_dict['main_module'].train()
    print("[Epoch: {}/{}] Loss: {:.4g}, PSNR: {:.4g}, SSIM: {:.4g}".format(inps_dict['t']+1, args.max_iter, test_loss, test_psnr, test_ssim))
    
def write_freq_actions(inps_dict, preruni_dict):
    args = inps_dict['args']
    end_time = time.time()
    
    preruni_dict['main_module'].eval()
    test_psnr, test_ssim, test_loss = preruni_dict['main_module'].test_psnr_ssim()
    preruni_dict['main_module'].train()
    
    # for optim in preruni_dict['main_module'].optims:
    #     for p in optim.param_groups:
    #         lr__ = p['lr']
    #         break
    
    r_logs = open(os.path.join(inps_dict['res_dir'], 'logs_r{}.txt'.format(inps_dict['run_number'])), "a")
    r_logs.write('Epoch: {}/{}, Time: {:}, Loss: {:.4f}, '.format(inps_dict['t']+1,args.max_iter,end_time-inps_dict['start_time'],inps_dict['losses_r'][-1]))
    r_logs.write("PSNR: {:.4g} | SSIM: {:.4g}\n".format(test_psnr, test_ssim))
    r_logs.close()
    plot_change_of_value(inps_dict['psnrs_r'], 'PSNR', inps_dict['repr_str'], inps_dict['run_number'], to_save=True, save_folder=inps_dict['res_dir'])
    plot_change_of_value(inps_dict['ssims_r'], 'SSIM', inps_dict['repr_str'], inps_dict['run_number'], to_save=True, save_folder=inps_dict['res_dir'])
    plot_change_of_value(inps_dict['losses_r'], 'Loss', inps_dict['repr_str'], inps_dict['run_number'], to_save=True, save_folder=inps_dict['res_dir'])

def gif_freq_actions(inps_dict, preruni_dict, output_im=None):
    args = inps_dict['args']
    preruni_dict['main_module'].eval()
    ep_no = inps_dict['t']+1
    if output_im is None:
        output_im, test_psnr , test_ssim, test_loss = preruni_dict['main_module'].test_psnr_ssim(ret_im=True)
        output_im=output_im.cpu().detach().numpy().squeeze() 
        print("[Epoch: {}/{}] Loss: {:.4g}, PSNR: {:.4g}, SSIM: {:.4g}".format(inps_dict['t']+1, args.max_iter, test_loss, test_psnr, test_ssim))
    else:
        im_nm = glob.glob(os.path.join(inps_dict['res_dir'], 'rec_*'))[0]
        ep_no = int(im_nm[im_nm.find('ep')+2:im_nm.find('ep')+im_nm[im_nm.find('ep'):].find('_')])
    preruni_dict['main_module'].train()
    #
    for ax_cr_sg in [0,1,2]:
        plot_max_mse=True
        #p = multiprocessing.Process(target = make_gif_of_rec_vs_gt.main, args=(output_im.cpu().detach().numpy().squeeze(), preruni_dict['main_module'].image.cpu().detach().numpy().squeeze(), 350, ax_cr_sg, args.pt, inps_dict['res_dir'], args, inps_dict['repr_str'], plot_max_mse,))
        #p.start()
        make_gif_of_rec_vs_gt.main(output_im, preruni_dict['main_module'].image.cpu().detach().numpy().squeeze(), 350, ax_cr_sg, inps_dict['res_dir'], args, inps_dict['repr_str'], ep_no, plot_max_mse)
    
       
    
def postrun_i_actions(inps_dict, preallruns_dict, preruni_dict):
    args = inps_dict['args']

    preruni_dict['model'].eval()
    with torch.no_grad():
        deformed_grid = preruni_dict['grid'] + (preruni_dict['model_tr'](preruni_dict['encoder_tr'].embedding(preruni_dict['grid'])))  # [B, C, H, W, 1]
        output_im = preruni_dict['model'](preruni_dict['encoder'].embedding(deformed_grid))
        output_im = output_im.reshape((1,128,128,64,1))
        test_loss = preruni_dict['mse_loss_fn'](output_im, preruni_dict['image'])
        test_psnr = - 10 * torch.log10(test_loss).item()
        print('MODEL PSNR: {:.5f}'.format(test_psnr))
        test_loss = test_loss.item()

    # train_writer.add_scalar('test_loss', test_loss, iterations + 1)
    # train_writer.add_scalar('test_psnr', test_psnr, iterations + 1)
    # Must transfer to .cpu() tensor firstly for saving images
    #save_image_3d(test_output, preruni_dict['slice_idx'], os.path.join(preruni_dict['image_directory'], "recon_{}_{:.4g}dB.png".format(inps_dict['t']+1, test_psnr)))
    
    r_logs = open(os.path.join(inps_dict['save_folder'], 'logs_r{}.txt'.format(inps_dict['run_number'])), "a")
    r_logs.write('Epoch: {}, Train Loss: {:.4f}\n'.format(inps_dict['t']+1,inps_dict['losses_r'][-1]))
    to_write = "[Validation Iteration: {}/{}] Test loss: {:.4g} | Test psnr: {:.4g}".format(inps_dict['t']+1, args.max_iter, test_loss, test_psnr)
    r_logs.write(to_write)
    r_logs.close()
    #plt_model.plot_change_of_objective(inps_dict['f_zks_r'], args.obj, args.K, args.N, inps_dict['run_number'], True, inps_dict['save_folder'])
    
    
    #vals, metric_name, repr_str, runno, to_save=False, save_folder=None):
    
    print('**************')
    print('FINAL RES.: ' + to_write)
    preallruns_dict['mean_psnr'] += test_psnr/args.noRuns
    print('******************************************')
    
    preallruns_dict['final_psnrs'][inps_dict['run_number']] = test_psnr
    np.save(os.path.join(inps_dict['save_folder'],'final_psnrs'), preallruns_dict['final_psnrs'])
    
    np.save(os.path.join(inps_dict['save_folder'],'psnrs_{}'.format(inps_dict['run_number'])), inps_dict['psnrs_r'])
    
    main_logs = open(os.path.join(inps_dict['save_folder'], 'main_logs.txt'), "a")
    main_logs.write(to_write)
    main_logs.close()
    
    print('Encoder B[65:68,:]: ',preruni_dict['encoder'].B[65:68,:])
    

def postallruns_actions(inps_dict, preallruns_dict):
    args =inps_dict['args']
    save_folder = inps_dict['save_folder']
    final_psnrs = preallruns_dict['final_psnrs']
    print("All psnr values", final_psnrs)
    main_log = 'Mean psnr over all runs: {:.6f} \n'.format(preallruns_dict['mean_psnr'])
    print(main_log)
    print('********************************TRAINING ENDED**********************************************')
    main_logs = open(os.path.join(save_folder, 'main_logs.txt'), "a")
    main_logs.write(main_log)
    main_logs.close()
    
