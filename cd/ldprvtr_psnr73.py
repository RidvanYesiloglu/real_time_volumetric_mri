# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 02:45:03 2022

@author: ridva
"""
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

from torchnufftexample import create_radial_mask, project_radial, backproject_radial
from networks import Positional_Encoder, FFN, SIREN
from skimage.metrics import structural_similarity as ssim
import torch.nn as nn
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
def update_runset_summary(args, runset_folder):
    reads = ""
    for i in range(args.totalInds):
        if Path(os.path.join(runset_folder, 'ind_runs', 'run{}.txt'.format(i+1))).is_file():
            run_i_file = open(os.path.join(runset_folder, 'ind_runs', 'run{}.txt'.format(i+1)), "r+")
            reads += run_i_file.read() + "\n"
            run_i_file.close()
    summary_file = open(os.path.join(runset_folder, 'runset_{}.txt'.format(args.runsetName)), "w")
    summary_file.write(reads)
    summary_file.close()

def main(args=None, im_ind=None):
    dtype = torch.float
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.set_printoptions(precision=7)
    args = Namespace(net = {"network_input_size":512,'network_width':150,'network_depth':5,'network_output_size':3}, \
                     encoder = {'embedding':'gauss','scale':3, 'embedding_size':256, 'coordinates_size':3}, \
                         img_path='../data/patient19/volume_1739.npy')#, gpu_id=int(input('Enter GPU ID: ')))
    des_n_proj = int(input('Desired nproj or -1 if prior: '))
    start_ind = 1
    end_ind = 29
    ptno = int(input('Enter patient no: '))
    runname = (input('Enter runname: '))
    psnrs = np.zeros((end_ind-start_ind+1))
    ssims = np.zeros((end_ind-start_ind+1))
    img_path = '/raid/yesiloglu/data_4DMRI/pt_{}_5min/ims_tog.npy'.format(ptno)  #'/raid/yesiloglu/data_4DMRI/data73/ims_tog.npy'
    psnrs_p = np.zeros((end_ind-start_ind+1))
    ssims_p = np.zeros((end_ind-start_ind+1))
    image_pr = torch.from_numpy(np.expand_dims(np.load(img_path)[0],(0,-1)).astype('float32'))#.cuda(args.gpu_id)
    for im_ind in range(start_ind,end_ind+1):
        print('im ind: {}'.format(im_ind))
        # args.net['network_width']=150
        # args.net['network_depth']=5
        # args.net['network_output_size']=3
        # # Setup input encoder:
        # encoder_tr = Positional_Encoder(args)
        # # Setup model as SIREN
        # model_tr = SIREN(args.net)
        # #print(model_tr)
        # model_tr = model_tr.cuda(args.gpu_id)
        # args.net['network_width']=256
        # args.net['network_depth']=8
        # args.net['network_output_size']=1
        
        # # Setup input encoder:
        # encoder = Positional_Encoder(args)
        # # Setup model as SIREN
        # model = SIREN(args.net)
        # model = model.cuda(args.gpu_id)
        # model_tr = nn.DataParallel(model_tr,[args.gpu_id,3])
        # model = nn.DataParallel(model, [args.gpu_id, 3])
        # Setup loss function
        mse_loss_fn = torch.nn.MSELoss()
        if des_n_proj != -1:
            a_dir = '../cascade_models/load_prev_net/detailed_results/all_{}/vol_{}'.format(ptno, im_ind) #'../cascade_models/detailed_results/all_data73/vol_{}'.format(im_ind)
            for i in range(1):#3
                kk = [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name)) and name.endswith(runname)]
                a_dir += '/'+kk[0]
            model_list = [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]
            # for name in model_list:
            #     npr_ind = name.find('nproj_')
            #     found_nproj = int(name[npr_ind+6:name.find('&', npr_ind)])
            #     if found_nproj == des_n_proj:
            #         a_dir += '/'+name
            print(a_dir)
            print(os.listdir(a_dir))
        #     model_path = a_dir + '/' + ([name for name in os.listdir(a_dir) if name.endswith('.pt')][0])
        #     state_dict = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(args.gpu_id))
        #     model.load_state_dict(state_dict['net'])
        #     encoder.B = state_dict['enc'].cuda(args.gpu_id)
        #     #model = model.cuda(args.gpu_id)
        #     model_tr.load_state_dict(state_dict['net_tr'])
        #     encoder_tr.B = state_dict['enc_tr'].cuda(args.gpu_id)
        #     #model_tr = model_tr.cuda(args.gpu_id)
            
        # #print('Load pretrain model: {}'.format(model_path))
        
        # # Setup data loader
        # #print('Load image: {}'.format(args.img_path))
        # #data_loader = get_data_loader(config['data'], config['img_path'], config['img_size'], img_slice=None, train=True, batch_size=config['batch_size'])
        
        # # Input coordinates (x, y, z) grid and target image
        # #grid = grid.cuda()  # [bs, c, h, w, 3], [0, 1]
        # #image = image.cuda()  # [bs, c, h, w, 1], [0, 1]
        
        image = torch.from_numpy(np.expand_dims(np.load(img_path)[im_ind],(0,-1)).astype('float32'))#.cuda(args.gpu_id)
        #image = np.expand_dims(np.load(img_path)[im_ind],(0,-1)).astype('float32')#.cuda(args.gpu_id)
        # np.save('imagg246', image.numpy())
        # np.save('imagg0_19', torch.from_numpy(np.expand_dims(np.load(img_path)[0],(0,-1)).astype('float32')))
        # grid_np = np.asarray([(x,y,z) for x in range(128) for y in range(128) for z in range(64)]).reshape((1,128,128,64,3))
        # grid_np = (grid_np/np.array([128,128,64])) + np.array([1/256.0,1/256.0,1/128.0])
        # grid = torch.from_numpy(grid_np.astype('float32')).cuda(args.gpu_id)
        # # print('Image min: {} and max: {}'.format(image.min(), image.max()))
        # # print('Image and grid created. Image shape: {}, grid shape: {}'.format(image.shape, grid.shape))
        # # print('Image element size: {} and neleements: {}, size in megabytes: {}'.format(image.element_size(), image.nelement(), (image.element_size()*image.nelement())/1000000.0))
        # # print('Grid element size: {} and neleements: {}, size in megabytes: {}'.format(grid.element_size(), grid.nelement(), (grid.element_size()*grid.nelement())/1000000.0))

        # # Data loading
        # test_data = (grid, image)
        # test_embedding = encoder.embedding(test_data[0])
       
        # with torch.no_grad():
        #     deformed_grid = grid + (model_tr(encoder_tr.embedding(grid)))  # [B, C, H, W, 1]
        #     #deformed_grid = preruni_dict['grid'] + (preruni_dict['model_tr'](preruni_dict['train_embedding_tr']))  # [B, C, H, W, 1]
        #     output_im = model(encoder.embedding(deformed_grid))
        #     output_im = output_im.reshape((1,128,128,64,1))
        #     test_loss = mse_loss_fn(output_im, image)
        #     test_psnr = - 10 * torch.log10(test_loss).item()
        #     #print('MODEL PSNR: {:.5f}'.format(test_psnr))
            
        #     test_loss = test_loss.item()
        #     test_ssim = ssim(output_im.cpu().numpy().squeeze(), image.cpu().numpy().squeeze(), data_range=1)
        # #np.save(os.path.join(inps_dict['save_folder'], 'pretrainmodel_out'), test_output.detach().cpu().numpy())
        output_im = a_dir + '/' + ([name for name in os.listdir(a_dir) if name.startswith('savedrec')][0])
        print('rec: ', output_im)
        output_im = torch.from_numpy(np.load(output_im))
        print(output_im.shape, image.shape)
        #output_im = torch.from_numpy(np.expand_dims(np.load(img_path)[0],(0,-1)).astype('float32'))#.cuda(args.gpu_id)
        #output_im = np.expand_dims(np.load(img_path)[0],(0,-1)).astype('float32')#.cuda(args.gpu_id)
        
        test_psnr = - 10 * torch.log10(mse_loss_fn(output_im, image))
        
        test_psnr_pr = - 10 * torch.log10(mse_loss_fn(image, image_pr))
        #print('MODEL PSNR: {:.5f}'.format(test_psnr))
        #np.save('recc246', output_im)
        #test_loss = test_loss.item()
        test_ssim = ssim(output_im.detach().cpu().numpy().squeeze(), image.detach().cpu().numpy().squeeze(), data_range=1)
        test_ssim_pr = ssim(image_pr.detach().cpu().numpy().squeeze(), image.detach().cpu().numpy().squeeze(), data_range=1)
        if im_ind %3==0:
            print(test_psnr, test_ssim)
        psnrs[im_ind-start_ind] = test_psnr
        ssims[im_ind-start_ind] = test_ssim
        
        psnrs_p[im_ind-start_ind] = test_psnr_pr
        ssims_p[im_ind-start_ind] = test_ssim_pr
        if im_ind % 10 == 0:
            # np.save('psnr_data73', psnrs)#
            # np.save('ssim_data73', ssims)#
            # np.save('psnr_p_data73', psnrs_p)#
            # np.save('ssim_p_data73', ssims_p)#
            np.save('psnr_ldprvnt_pt{}'.format(ptno), psnrs)
            np.save('ssim_ldprvnt_pt{}'.format(ptno), ssims)
            np.save('psnr_p_pt{}'.format(ptno), psnrs_p)
            np.save('ssim_p_pt{}'.format(ptno), ssims_p)
    #print('ALL PSNRS: ')
    #print(psnrs)
    print('Average: {:.5f}, min: {:.5f} (at {}), max: {:.5f} (at {})'.format(psnrs.mean(), psnrs.min(), psnrs.argmin()+start_ind, psnrs.max(), psnrs.argmax()+start_ind))
    #print('ALL SSIMS: ')
    #print(ssims)
    print('Average: {:.5f}, min: {:.5f} (at {}), max: {:.5f} (at {})'.format(ssims.mean(), ssims.min(), ssims.argmin()+start_ind, ssims.max(), ssims.argmax()+start_ind))
    # np.save('psnr_data73', psnrs)#
    # np.save('ssim_data73', ssims)#
    # np.save('psnr_p_data73', psnrs_p)#
    # np.save('ssim_p_data73', ssims_p)#
    np.save('psnr_pt{}'.format(ptno), psnrs)
    np.save('ssim_pt{}'.format(ptno), ssims)
    np.save('psnr_p_pt{}'.format(ptno), psnrs_p)
    np.save('ssim_p_pt{}'.format(ptno), ssims_p)
if __name__ == "__main__":
    main() 
