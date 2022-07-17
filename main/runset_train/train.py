import runset_train.parameters as parameters
import torch
import torch.optim as optim
import torch.distributions.bernoulli as Bernoulli
import numpy as np
import math
import os
import errno
from tqdm import tqdm #(for time viewing)
import time
import models.nerp.write_actions_nerp as wr_acts
from pathlib import Path

import torch.backends.cudnn as cudnn
from utils import mri_fourier_transform_3d, save_image_3d, PSNR, check_gpu

import glob
from jacobian import JacobianReg
from torchnufftexample import create_radial_mask, project_radial, backproject_radial
import pickle


#working_dir = '/home/yesiloglu/projects/cascaded_nerp/cascade_models'
def main(args=None):
    # Get arguments
    params_dict = parameters.decode_arguments_dictionary('params_dictionary')
    if args is None:
        args = parameters.get_arguments(params_dict)
    # Create representative string for the training
    repr_str = parameters.create_repr_str(args, [info.name for info in params_dict.param_infos], wantShort=True, params_dict=params_dict)
    long_repr_str = parameters.create_repr_str(args, [info.name for info in params_dict.param_infos], wantShort=False, params_dict=params_dict)
    print(f'Representative string for the training is: {repr_str}')
    
    # Constants
    pt_dir = f'/home/yesiloglu/projects/real_time_volumetric_mri/results/{args.pt}/'
    res_dir = f'{pt_dir}{args.conf}/t_{args.im_ind}/{repr_str}'
    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.set_printoptions(precision=7)
    print_freq = 500 # print results once in "print_frequency" epochs 
    write_freq = 1000 # print results once in "print_frequency" epochs
    gif_freq = 1000
    cudnn.benchmark = True
    
    if not os.path.exists(res_dir):
        try:
            os.makedirs(res_dir)
        except:
            raise

    for run_number in range(args.noRuns):
        torch.cuda.empty_cache()

        
        preruni_dict = wr_acts.prerun_i_actions({'res_dir': res_dir, 'args':args, 'repr_str':repr_str, 'long_repr_str':long_repr_str, 'run_number':run_number, 'device':device, 'dtype':dtype})

        psnrs_r = [] 
        ssims_r = [] 
        losses_r = []
        #l_components_r = []
        start_time = time.time()
        
        
        for ep in tqdm(range(args.max_iter)):
            preruni_dict['main_module'].train()
            for optimizer in preruni_dict['main_module'].optims:
                optimizer.zero_grad()
            train_loss = preruni_dict['main_module']()
            train_loss.backward()
            
            optimizer.step()
                
            test_psnr , test_ssim, test_loss = preruni_dict['main_module'].test_psnr_ssim()
            losses_r.append(test_loss)
            psnrs_r.append(test_psnr)
            ssims_r.append(test_ssim)
            
            #'pri_emb','trn_wo_trns','trn_w_trns'
            # Save the model and the reconstruction
            if ((args.conf == 'pri_emb') and (test_psnr == max(psnrs_r))):
                #wr_acts.write_freq_actions({'args':args, 't':ep, 'start_time':start_time, 'res_dir': res_dir, 'run_number':run_number,\
                #    'repr_str':repr_str,'psnrs_r':psnrs_r,'ssims_r':ssims_r,'losses_r':losses_r}, preruni_dict)
            
                # Save the test output and the model:
                output_im, test_psnr , test_ssim, test_loss = preruni_dict['main_module'].test_psnr_ssim(ret_im=True)
                for filename in glob.glob(os.path.join(res_dir, 'md_{}*'.format(repr_str))):
                    os.remove(filename)
                for filename in glob.glob(os.path.join(res_dir, 'rec_{}*'.format(repr_str))):
                    os.remove(filename)
                model_name = os.path.join(res_dir, 'md_{}_ep{}_{:.4g}dB.pt'.format(repr_str, ep+1, test_psnr))
                torch.save(preruni_dict['main_module'].get_to_save_dict(), model_name)
                np.save(os.path.join(res_dir,'rec_{}_ep{}_{:.4g}dB'.format(repr_str, ep+1, test_psnr)), output_im.detach().cpu().numpy())
                np.save(os.path.join(res_dir,'psnrs_r{}'.format(run_number)), np.asarray(psnrs_r))
                np.save(os.path.join(res_dir,'ssims_r{}'.format(run_number)), np.asarray(ssims_r))
                np.save(os.path.join(res_dir,'losses_r{}'.format(run_number)), np.asarray(losses_r))
            elif ((args.conf != 'pri_emb') and ((test_psnr == max(psnrs_r)) or ((ep+1)%50==0))):
                #wr_acts.write_freq_actions({'args':args, 't':ep, 'start_time':start_time, 'res_dir': res_dir, 'run_number':run_number,\
                #    'repr_str':repr_str,'psnrs_r':psnrs_r,'ssims_r':ssims_r,'losses_r':losses_r}, preruni_dict)
            
                # Save the test output and the model:
                output_im, test_psnr , test_ssim, test_loss = preruni_dict['main_module'].test_psnr_ssim(ret_im=True)
                if ((ep+1)%50==0):
                    model_name = os.path.join(res_dir, 'dd_md_{}_ep{}_{:.4g}dB.pt'.format(repr_str, ep+1, test_psnr))
                    torch.save(preruni_dict['main_module'].get_to_save_dict(), model_name)
                    np.save(os.path.join(res_dir,'dd_rec_{}_ep{}_{:.4g}dB'.format(repr_str, ep+1, test_psnr)), output_im.detach().cpu().numpy())
                if (test_psnr == max(psnrs_r)):
                    for filename in glob.glob(os.path.join(res_dir, 'md_{}*'.format(repr_str))):
                        os.remove(filename)
                    for filename in glob.glob(os.path.join(res_dir, 'rec_{}*'.format(repr_str))):
                        os.remove(filename)
                    model_name = os.path.join(res_dir, 'md_{}_ep{}_{:.4g}dB.pt'.format(repr_str, ep+1, test_psnr))
                    torch.save(preruni_dict['main_module'].get_to_save_dict(), model_name)
                    np.save(os.path.join(res_dir,'rec_{}_ep{}_{:.4g}dB'.format(repr_str, ep+1, test_psnr)), output_im.detach().cpu().numpy())
                np.save(os.path.join(res_dir,'psnrs_r{}'.format(run_number)), np.asarray(psnrs_r))
                np.save(os.path.join(res_dir,'ssims_r{}'.format(run_number)), np.asarray(ssims_r))
                np.save(os.path.join(res_dir,'losses_r{}'.format(run_number)), np.asarray(losses_r)) 
            # if (args.conf == 2) and (test_psnr == max(psnrs_r)): # network training
            #     # Save the test output and the model:
            #     for filename in glob.glob(os.path.join(save_folder, 'savedmodel_run{}*'.format(run_number))):
            #         os.remove(filename)
            #     for filename in glob.glob(os.path.join(save_folder, 'savedrec_run{}*'.format(run_number))):
            #         os.remove(filename)
            #     #np.save(os.path.join(save_folder,'defpr_run{}_ep{}_{:.4g}dB'.format(run_number, t+1, test_psnr)), deformed_prior.detach().cpu().numpy())
            #     model_name = os.path.join(save_folder, 'savedmodel_run{}_ep{}_{:.4g}dB.pt'.format(run_number, t+1, test_psnr))
            #     torch.save({'net_tr': preruni_dict['model_tr'].state_dict(), \
            #             'enc_tr': preruni_dict['encoder_tr'].B, \
            #             'opt_tr': preruni_dict['optim_tr'].state_dict(), \
            #             'net': preruni_dict['model'].state_dict(), \
            #             'enc': preruni_dict['encoder'].B, \
            #             'opt': preruni_dict['optim'].state_dict()}, \
            #             model_name)
            #     np.save(os.path.join(save_folder,'savedrec_run{}_ep{}_{:.4g}dB'.format(run_number, t+1, test_psnr)), output_im.detach().cpu().numpy())
            #     # with open('l_comps_{}'.format(run_number), 'wb') as f:
            #     #     pickle.dump(l_components_r, f)
            # elif (args.conf == 1) and (test_psnr == max(psnrs_r)): # prior embedding
            #     # Save the test output and the model:
            #     for filename in glob.glob(os.path.join(res_dir, 'savedmodel_run{}*'.format(run_number))):
            #         os.remove(filename)
            #     for filename in glob.glob(os.path.join(res_dir, 'savedrec_run{}*'.format(run_number))):
            #         os.remove(filename)
            #     #np.save(os.path.join(save_folder,'defpr_run{}_ep{}_{:.4g}dB'.format(run_number, t+1, test_psnr)), deformed_prior.detach().cpu().numpy())
            #     model_name = os.path.join(res_dir, 'savedmodel_run{}_ep{}_{:.4g}dB.pt'.format(run_number, t+1, test_psnr))
            #     # torch.save({'net_tr': preruni_dict['model_tr'].state_dict(), \
            #     #         'enc_tr': preruni_dict['encoder_tr'].B, \
            #     #         'opt_tr': preruni_dict['optim_tr'].state_dict(), \
            #     #         'net': preruni_dict['model'].state_dict(), \
            #     #         'enc': preruni_dict['encoder'].B, \
            #     #         'opt': preruni_dict['optim'].state_dict()}, \
            #     #         model_name)
            #     torch.save({'net': preruni_dict['model'].state_dict(), \
            #             'enc': preruni_dict['encoder'].B, \
            #             'opt': preruni_dict['optim'].state_dict()}, \
            #             model_name)
            #     np.save(os.path.join(res_dir,'savedrec_run{}_ep{}_{:.4g}dB'.format(run_number, t+1, test_psnr)), output_im.detach().cpu().numpy())
            if ep % 100 == 0:
                np.save(os.path.join(res_dir,'psnrs_{}'.format(repr_str)), np.asarray(psnrs_r))
                np.save(os.path.join(res_dir,'ssims_{}'.format(repr_str)), np.asarray(ssims_r))
                np.save(os.path.join(res_dir,'losses_{}'.format(repr_str)), np.asarray(losses_r))
                # with open('l_comps_{}'.format(run_number), 'wb') as f:
                #     pickle.dump(l_components_r, f)
            # Print
            if (ep+1) % print_freq == 0:
                wr_acts.print_freq_actions({'args':args, 't':ep}, preruni_dict)
            if (ep+1) % write_freq == 0:   
                wr_acts.write_freq_actions({'args':args, 't':ep, 'start_time':start_time, 'res_dir': res_dir, 'run_number':run_number,\
                    'repr_str':repr_str,'psnrs_r':psnrs_r,'ssims_r':ssims_r,'losses_r':losses_r}, preruni_dict)
            if (ep+1) % gif_freq == 0:
                wr_acts.gif_freq_actions({'args':args, 't':ep, 'res_dir': res_dir, 'repr_str':repr_str}, preruni_dict)
                # Save final model
            # if (t + 1) % args.image_save_iter == 0:
            #     model_name = os.path.join(preruni_dict['checkpoint_directory'], 'model_%06d.pt' % (t + 1))
            #     torch.save({'net': preruni_dict['model'].state_dict(), \
            #                 'enc': preruni_dict['encoder'].B, \
            #                 'opt': preruni_dict['optim'].state_dict(), \
            #                 }, model_name)
        wr_acts.write_freq_actions({'args':args, 't':ep, 'start_time':start_time, 'res_dir': res_dir, 'run_number':run_number,\
                    'repr_str':repr_str,'psnrs_r':psnrs_r,'ssims_r':ssims_r,'losses_r':losses_r}, preruni_dict)
        wr_acts.gif_freq_actions({'args':args, 't':ep, 'res_dir': res_dir, 'repr_str':repr_str}, preruni_dict)
        #wr_acts.postrun_i_actions({'args':args, 'run_number':run_number, 'save_folder':save_folder, 'losses_r':losses_r, 'psnrs_r':psnrs_r,'device':device, \
         #                         't':t},preallruns_dict, preruni_dict)

    #wr_acts.postallruns_actions({'args':args, 'save_folder':save_folder}, preallruns_dict)
    
    # mydicti = globals()
    # for name in mydicti:
    #     print(name)
    #     if not name.startswith('_'):
    #         del globals()[name]
    # with torch.cuda.device('cuda:2'):
    #     torch.cuda.empty_cache()
    # with torch.cuda.device('cuda:3'):
    #     torch.cuda.empty_cache()
if __name__ == "__main__":
    main() 
