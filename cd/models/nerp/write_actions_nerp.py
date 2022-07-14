import os
import torch
import numpy as np

import time

from models.nerp.model_nerp import Main_Module as Main_Module
from models.nerp.plot_nerp import plot_change_of_objective


from utils import prepare_sub_folder, mri_fourier_transform_3d, complex2real, random_sample_uniform_mask, random_sample_gaussian_mask, save_image_3d, PSNR, check_gpu

from torchnufftexample import create_radial_mask, project_radial, backproject_radial
import sys
import torch.nn as nn

# inps_dict: {'save_folder': save_folder, 'args':args, 'repr_str':repr_str, 'run_number':run_number, 'model': model}
# outs_dict: 
def prerun_i_actions(inps_dict):
    args =inps_dict['args']
    main_module = Main_Module(args)
    main_module.eval()
    test_psnr, test_ssim, test_loss = main_module.test_psnr_ssim()
    main_module.train()
    
    init_psnr_str = 'Initial loss: {:.4f}, initial psnr: {:.4f}, initial ssim: {:.4f}\n'.format(test_loss, test_psnr, test_ssim)
    
    r_logs = open(os.path.join(inps_dict['res_dir'], 'logs_{}_{}.txt'.format(inps_dict['run_number'], inps_dict['repr_str'])), "a")
    r_logs.write('Runset Name: {}, Individual Run No: {}\n'.format(args.runsetName, args.indRunNo))
    r_logs.write('Configuration: {}\n'.format(inps_dict['repr_str']))
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
    
    r_logs = open(os.path.join(inps_dict['res_dir'], 'logs_{}_{}.txt'.format(inps_dict['run_number'], inps_dict['repr_str'])), "a")
    r_logs.write('Epoch: {}/{}, Time: {}, Loss: {:.4f}\n'.format(inps_dict['t']+1,args.max_iter,end_time-inps_dict['start_time'],inps_dict['losses_r'][-1]))
    to_write = "PSNR: {:.4g} | SSIM: {:.4g}\n".format(test_psnr, test_ssim)
    r_logs.write(to_write)
    start_time = time.time()
    r_logs.close()
    print('runnumber ', inps_dict['run_number'])
    inps_dict['repr_str'] = 'asd'
    plot_change_of_objective(inps_dict['psnrs_r'], 'PSNR', inps_dict['repr_str'], inps_dict['run_number'], to_save=True, save_folder=inps_dict['res_dir'])
    plot_change_of_objective(inps_dict['ssims_r'], 'SSIM', inps_dict['repr_str'], inps_dict['run_number'], to_save=True, save_folder=inps_dict['res_dir'])
    plot_change_of_objective(inps_dict['losses_r'], 'Loss', inps_dict['repr_str'], inps_dict['run_number'], to_save=True, save_folder=inps_dict['res_dir'])
    print(to_write)
    return {'start_time':start_time}


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
    
    r_logs = open(os.path.join(inps_dict['save_folder'], 'logs_{}.txt'.format(inps_dict['run_number'])), "a")
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
    
