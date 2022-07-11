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
def main(args=None, im_ind=None):
    # Get arguments
    params_dict = parameters.decode_arguments_dictionary('params_dictionary')
    if args is None:
        args = parameters.get_arguments(params_dict)
    # Create representative string for the training
    repr_str = parameters.create_repr_str(args, [info.name for info in params_dict.param_infos], wantShort=True, params_dict=params_dict)
    print(f'Representative string for the training is: {repr_str}')
    
    
    # Constants
    pt_dir = f'/home/yesiloglu/projects/cascaded_nerp/results/{args.pt}/'
    prior_dir = pt_dir + 'prior_model/'
    res_dir = (pt_dir + 'pri_emb/') if (args.prEmOrTr == 1) else (pt_dir + 'net_trn/')
    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.set_printoptions(precision=7)
    print_freq = 100 # print results once in "print_frequency" epochs 
    write_freq = 100 # print results once in "print_frequency" epochs
    cudnn.benchmark = True
    
    if not os.path.exists(res_dir):
        try:
            os.makedirs(res_dir)
        except:
            raise

    for run_number in range(args.noRuns):
        torch.cuda.empty_cache()

        
        preruni_dict = wr_acts.prerun_i_actions({'res_dir': res_dir, 'args':args, 'repr_str':repr_str, 'run_number':run_number, 'device':device, 'dtype':dtype})

        psnrs_r = [] 
        losses_r = []
        l_components_r = []
        start_time = time.time()
        
        # with torch.no_grad():
        #     deformed_grid = preruni_dict['grid'] + (preruni_dict['model_tr'](preruni_dict['encoder_tr'].embedding(preruni_dict['grid'])))  # [B, C, H, W, 1]
        #     #deformed_grid = preruni_dict['grid'] + (preruni_dict['model_tr'](preruni_dict['train_embedding_tr']))  # [B, C, H, W, 1]
        #     output_im = preruni_dict['model'](preruni_dict['encoder'].embedding(deformed_grid))
        #     output_im = output_im.reshape((1,128,128,64,1))
        #     test_loss = preruni_dict['mse_loss_fn'](output_im, preruni_dict['image'])
        #     test_psnr = - 10 * torch.log10(test_loss).item()
        #     print('STARTING MODEL PSNR: {:.5f}'.format(test_psnr))
        #torch.cuda.empty_cache()
        reg = JacobianReg(gpu_id=args.gpu_id) # Jacobian regularization
        #lambda_JR = 100 # hyperparameter
        print('after init psnr calc')
        check_gpu(args.gpu_id)
        for t in tqdm(range(args.max_iter)):
            # print('T=',t)
            #preruni_dict['model_tr'].train()
            #preruni_dict['optim_tr'].zero_grad()
            preruni_dict['model'].train()
            preruni_dict['optim'].zero_grad()
            # print('before')
            # check_gpu(args.gpu_id)
            
            #deformed_grid = preruni_dict['grid'] + (preruni_dict['model_tr'](preruni_dict['encoder_tr'].embedding(preruni_dict['grid'])))  # [B, C, H, W, 1]
            #deformed_grid = preruni_dict['grid'] + (preruni_dict['model_tr'](preruni_dict['train_embedding_tr']))  # [B, C, H, W, 1]
            
            #print('shapes ', deformed_grid.shape, ' grid: ', preruni_dict['grid'].shape)
            # print('after 1')
            # check_gpu(args.gpu_id)
            # torch.cuda.empty_cache()
            # print('after 2')
            # check_gpu(args.gpu_id)
            
            output_im = preruni_dict['model'](preruni_dict['encoder'].embedding(preruni_dict['grid']))
            output_im = output_im.reshape((1,128,128,64,1))
            output_im.retain_grad()
            # print('after 3')
            # check_gpu(args.gpu_id)
            #train_kdata = project_radial(output_im, preruni_dict['ktraj'], preruni_dict['im_size'], preruni_dict['grid_size'])
            #train_spec = mri_fourier_transform_3d(train_output)  # [B, H, W, C]
            #train_spec = train_spec * preruni_dict['mask'][None, ..., None]
            #print('dumb tensors 1')
            #dump_tensors()
            #torch.cuda.empty_cache()
            #print('dumb tensors 2')
            #dump_tensors()
            #Reg_loss = reg(preruni_dict['grid'], deformed_grid)   # Jacobian regularization
            #main_loss = preruni_dict['mse_loss_fn'](train_kdata, preruni_dict['image_kdata'])
            
            mse_loss_fn = torch.nn.MSELoss()
            # gl_x = mse_loss_fn(output_im.narrow(1,1,127), output_im.narrow(1,0,127))
            # gl_y = mse_loss_fn(output_im.narrow(2,1,127), output_im.narrow(2,0,127))
            # gl_z = mse_loss_fn(output_im.narrow(3,1,63), output_im.narrow(3,0,63))
            # gl_t = mse_loss_fn(output_im, preallruns_dict['prev_rec'])
            # geometric_loss = (gl_x + gl_y + gl_z + gl_t)/4.0
            print('gt im shape ' + preruni_dict['image'])
            train_loss = preruni_dict['mse_loss_fn'](output_im, preruni_dict['image'])
            #main_loss+ args.lambda_JR*Reg_loss + args.lambda_gl*geometric_loss # full loss 
            #print('R: ', R, ' main_loss: ', main_loss)
            train_loss.backward()
            # print('after i backward')
            # check_gpu(args.gpu_id)
            preruni_dict['optim_tr'].step()
            preruni_dict['optim'].step()
            # Add loss to the losses list for r
            losses_r.append(train_loss.item())
            l_components_r.append([main_loss.item(), Reg_loss.item(), geometric_loss.item()])
            
            with torch.no_grad():
                test_loss = preruni_dict['mse_loss_fn'](output_im, preruni_dict['image'])
                test_psnr = - 10 * torch.log10(test_loss).item()
                # print('STARTING MODEL PSNR: {:.5f}'.format(test_psnr))

            psnrs_r.append(test_psnr)
            if (args.priEmOrTra == 2) and (test_psnr == max(psnrs_r)): # network training
                # Save the test output and the model:
                for filename in glob.glob(os.path.join(save_folder, 'savedmodel_run{}*'.format(run_number))):
                    os.remove(filename)
                for filename in glob.glob(os.path.join(save_folder, 'savedrec_run{}*'.format(run_number))):
                    os.remove(filename)
                #np.save(os.path.join(save_folder,'defpr_run{}_ep{}_{:.4g}dB'.format(run_number, t+1, test_psnr)), deformed_prior.detach().cpu().numpy())
                model_name = os.path.join(save_folder, 'savedmodel_run{}_ep{}_{:.4g}dB.pt'.format(run_number, t+1, test_psnr))
                torch.save({'net_tr': preruni_dict['model_tr'].state_dict(), \
                        'enc_tr': preruni_dict['encoder_tr'].B, \
                        'opt_tr': preruni_dict['optim_tr'].state_dict(), \
                        'net': preruni_dict['model'].state_dict(), \
                        'enc': preruni_dict['encoder'].B, \
                        'opt': preruni_dict['optim'].state_dict()}, \
                        model_name)
                np.save(os.path.join(save_folder,'savedrec_run{}_ep{}_{:.4g}dB'.format(run_number, t+1, test_psnr)), output_im.detach().cpu().numpy())
                with open('l_comps_{}'.format(run_number), 'wb') as f:
                    pickle.dump(l_components_r, f)
            elif (args.priEmOrTra == 1) and (test_psnr == max(psnrs_r)): # prior embedding
                # Save the test output and the model:
                for filename in glob.glob(os.path.join(res_dir, 'savedmodel_run{}*'.format(run_number))):
                    os.remove(filename)
                for filename in glob.glob(os.path.join(res_dir, 'savedrec_run{}*'.format(run_number))):
                    os.remove(filename)
                #np.save(os.path.join(save_folder,'defpr_run{}_ep{}_{:.4g}dB'.format(run_number, t+1, test_psnr)), deformed_prior.detach().cpu().numpy())
                model_name = os.path.join(res_dir, 'savedmodel_run{}_ep{}_{:.4g}dB.pt'.format(run_number, t+1, test_psnr))
                # torch.save({'net_tr': preruni_dict['model_tr'].state_dict(), \
                #         'enc_tr': preruni_dict['encoder_tr'].B, \
                #         'opt_tr': preruni_dict['optim_tr'].state_dict(), \
                #         'net': preruni_dict['model'].state_dict(), \
                #         'enc': preruni_dict['encoder'].B, \
                #         'opt': preruni_dict['optim'].state_dict()}, \
                #         model_name)
                torch.save({'net': preruni_dict['model'].state_dict(), \
                        'enc': preruni_dict['encoder'].B, \
                        'opt': preruni_dict['optim'].state_dict()}, \
                        model_name)
                np.save(os.path.join(res_dir,'savedrec_run{}_ep{}_{:.4g}dB'.format(run_number, t+1, test_psnr)), output_im.detach().cpu().numpy())
            if t % 100 == 0:
                np.save(os.path.join(res_dir,'psnrs_r{}'.format(run_number)), np.asarray(psnrs_r))
                with open('l_comps_{}'.format(run_number), 'wb') as f:
                    pickle.dump(l_components_r, f)
            # Print
            if (t+1) % print_freq == 0:
                wr_acts.print_freq_actions({'args':args, 't':t, 'losses_r':losses_r})
            if (t+1) % write_freq == 0:   
                write_freq_dict = wr_acts.write_freq_actions({'args':args, 't':t, 'start_time':start_time, 'psnrs_r':psnrs_r, \
                    'res_dir': res_dir, 'run_number':run_number,'losses_r':losses_r}, preruni_dict)
                start_time = write_freq_dict['start_time']
                # Save final model
            if (t + 1) % args.image_save_iter == 0:
                model_name = os.path.join(preruni_dict['checkpoint_directory'], 'model_%06d.pt' % (t + 1))
                torch.save({'net': preruni_dict['model'].state_dict(), \
                            'enc': preruni_dict['encoder'].B, \
                            'opt': preruni_dict['optim'].state_dict(), \
                            }, model_name)
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
