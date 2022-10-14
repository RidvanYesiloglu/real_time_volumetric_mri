import os
import argparse
import shutil

import torch
import torch.nn as nn
import torchvision
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
import scipy.io as sio

import numpy as np
from tqdm import tqdm

from networks import Positional_Encoder, FFN, SIREN
from utils import get_config, prepare_sub_folder, get_data_loader_4d, save_image_3d, sample_cine_mask_GTV
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def loadoneimg(imgname, imgsize):
    image = np.fromfile(imgname,'float32')  # [C, H, W]
    image = np.reshape(image,imgsize)
    image = torch.tensor(image, dtype=torch.float32)[None, ...]  # [B, C, H, W]
    image = image / torch.max(image)  # [B, C, H, W], [0, 1]
    image = image.permute(1, 2, 3, 0)  # [C, H, W, 1]
    return image


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")

# Load experiment setting
opts = parser.parse_args()
config = get_config(opts.config)
max_iter = config['max_iter']

cudnn.benchmark = True
config['img_size'] = np.array([config['z_dim'],config['y_dim'],config['x_dim']])
config['prior_idArray'] = np.array([config['prior_id1'],config['prior_id2']])

# test sample info
start_id = config['test_idstart']
test_idArray = np.arange(config['test_idstart'],config['test_idend']+1)
libidArray = sio.loadmat(config['libinfo'])
libidArray = libidArray['libArray']


# Setup output folder
output_folder = os.path.splitext(os.path.basename(opts.config))[0]
model_name = 'img{}_{}_{}_{}_{}_{}_lr{:.2g}_encoder_{}'.format(config['img_size'], config['model'], \
        config['net']['network_input_size'], config['net']['network_width'], \
        config['net']['network_depth'], config['loss'], config['lr'], config['encoder']['embedding'])

if not(config['encoder']['embedding'] == 'none'):
    model_name += '_scale{}_size{}'.format(config['encoder']['scale'], config['encoder']['embedding_size'])


output_directory = os.path.join(opts.output_path + "/outputs", output_folder, config['data'], model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder


increment = 0.01/10
mask = sample_cine_mask_GTV(config['img_size'],config['GTV_1'],config['GTV_2']).cuda()


for img_id in test_idArray:

    # Setup input encoder:
    encoder = Positional_Encoder(config['encoder'])
    # Setup model
    if config['model'] == 'SIREN':
        model = SIREN(config['net'])
    elif config['model'] == 'FFN':
        model = FFN(config['net'])
    else:
        raise NotImplementedError
    model.cuda()

    # Setup optimizer
    if config['optimizer'] == 'Adam':
        optim = torch.optim.Adam(model.parameters(), lr=config['lr']/10, betas=(config['beta1'], config['beta2']), weight_decay=config['weight_decay'])
    else:
        NotImplementedError
    # Setup loss functions
    if config['loss'] == 'L2':
        loss_fn = torch.nn.MSELoss()
    elif config['loss'] == 'L1':
        loss_fn = torch.nn.L1Loss()
    else:
        NotImplementedError

    pretrain_path = config['pretrain_path']
    pretrain_model = os.path.join(pretrain_path,model_name,'checkpoints','model_001000.pt')
    #print(pretrain_model)
    state_dict = torch.load(pretrain_model)
    model.load_state_dict(state_dict['net'])
    encoder.B = state_dict['enc']
    model.eval()

    libid = libidArray[img_id-start_id,0]
    endpoint = libid*increment
    if(libid<0):
        output_id = 0
        grid_z,grid_y, grid_x, grid_t = torch.meshgrid([torch.linspace(0, 1, steps=config['z_dim']), \
                                            torch.linspace(0, 1, steps=config['y_dim']), \
                                            torch.linspace(0, 1, steps=config['x_dim']),\
                                            torch.linspace(endpoint,0,steps=len(config['prior_idArray']))])
    else:
        output_id = 1
        grid_z, grid_y, grid_x, grid_t = torch.meshgrid([torch.linspace(0, 1, steps=config['z_dim']), \
                                            torch.linspace(0, 1, steps=config['y_dim']), \
                                            torch.linspace(0, 1, steps=config['x_dim']),\
                                            torch.linspace(0,endpoint,steps=len(config['prior_idArray']))])
    grid = torch.stack([grid_z, grid_y, grid_x, grid_t], dim=-1)   
    grid = grid.cuda()  # [bs, c, h, w, 3], [0, 1]
    test_embedding = encoder.embedding(grid)
    test_output = model(test_embedding)
    train_embedding = encoder.embedding(grid)
    train_output = model(train_embedding)

    matFile = os.path.join(output_directory,'pretrain_'+str(img_id)+'.mat')
    sio.savemat(matFile,{'grid': grid_t.numpy(),'test':test_output.detach().cpu().numpy(),'train':train_output.detach().cpu().numpy()})


    image = loadoneimg(config['img_path'].format(img_id),config['img_size'])
    image_full = loadoneimg(config['img_path'].format(img_id),config['img_size']) # for recon quality eval
    image = image.cuda()
    image_full = image_full.cuda()
    image = image*mask[None, ..., None]
    

    #cor_slice = image[:,config['GTV_1'],:,:]
    #sag_slice = image[:,:,config['GTV_2'],:]
    #plt.imshow(cor_slice.detach().cpu().squeeze())
    #plt.show()
    #plt.imshow(sag_slice.detach().cpu().squeeze())
    #plt.show()
    

    for iterations in range(max_iter):
            model.train()
            optim.zero_grad()
    
            train_output = model(train_embedding)  # [B, C, H, W, 1]
            train_spec = train_output[:,:,:,output_id]  # [B, H, W, C]
            train_spec = train_spec * mask[None, ..., None]
            train_loss = 0.5 * loss_fn(train_spec, image)  # fit to sparse sample only, may add the two priors later
    
            train_loss.backward()
            optim.step()
    
            # Compute training psnr
            if (iterations + 1) % config['log_iter'] == 0:
                train_loss = train_loss.item()
                # train_writer.add_scalar('train_loss', train_loss, iterations + 1)
                print("[Iteration: {}/{}] Train loss: {:.4g} ".format(iterations + 1, max_iter, train_loss))


            if iterations == 0 or (iterations + 1) % config['val_iter'] == 0:
                model.eval()
                with torch.no_grad():
                    test_output = model(test_embedding)
                    #print('the shape of output',test_output.size())
                    test_spec = test_output[:,:,:,output_id]
                    test_spec = test_spec*mask[None, ..., None]
                    test_loss = 0.5 * loss_fn(test_spec, image)
                    test_psnr = - 10 * torch.log10(2 * test_loss).item()
                    test_loss = test_loss.item()

                    recon = test_output[:,:,:,output_id]
                    recon_loss = 0.5 * loss_fn(recon, image_full)
                    recon_psnr = - 10 * torch.log10(2 * recon_loss).item()
                    recon_loss = recon_loss.item()
                    print('recon shape ',recon.size())
                    test_ssim = ssim(recon.squeeze().cpu().numpy(), image_full.squeeze().cpu().numpy())

                    print("[Iteration: {}/{}] Validation psnr: {:.4g} Recon psnr: {:.4g} ".format(iterations + 1, max_iter, test_psnr, recon_psnr))
                    matFile = os.path.join(image_directory,'ret_img{}_iter{}.mat'.format(img_id,iterations))
                    #print('save mat',matFile)
                    
                    sio.savemat(matFile,{'test_output':test_output.detach().cpu().numpy(),'test_psnr':test_psnr,'recon_loss':recon_loss,'recon_psnr':recon_psnr,'test_ssim':test_ssim, 'output_id':output_id})



