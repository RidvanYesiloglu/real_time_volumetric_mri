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
from utils import get_config, prepare_sub_folder, get_data_loader_4d, save_image_3d


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")

# Load experiment setting
opts = parser.parse_args()
config = get_config(opts.config)
max_iter = config['max_iter']

cudnn.benchmark = True
config['img_size'] = np.array([config['z_dim'],config['y_dim'],config['x_dim']])
config['img_idArray'] = np.array([config['img_id1'],config['img_id2']])
print('img size info',np.shape(config['img_size']))

# Setup output folder
output_folder = os.path.splitext(os.path.basename(opts.config))[0]
model_name = os.path.join(output_folder, config['data'] + '/img{}_{}_{}_{}_{}_{}_lr{:.2g}_encoder_{}' \
    .format(config['img_size'], config['model'], \
        config['net']['network_input_size'], config['net']['network_width'], \
        config['net']['network_depth'], config['loss'], config['lr'], config['encoder']['embedding']))
if not(config['encoder']['embedding'] == 'none'):
    model_name += '_scale{}_size{}'.format(config['encoder']['scale'], config['encoder']['embedding_size'])
print(model_name)

output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder


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
model.train()

# Setup optimizer
if config['optimizer'] == 'Adam':
    optim = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']), weight_decay=config['weight_decay'])
else:
    NotImplementedError

# Setup loss functions
if config['loss'] == 'L2':
    loss_fn = torch.nn.MSELoss()
elif config['loss'] == 'L1':
    loss_fn = torch.nn.L1Loss()
else:
    NotImplementedError


# Setup data loader
print('Load image: {}'.format(config['img_path']))
data_loader = get_data_loader_4d(config['data'], config['img_path'], config['img_size'], config['img_idArray'], train=True, batch_size=config['batch_size'])

config['img_size'] = (config['img_size'], config['img_size'], config['img_size']) if type(config['img_size']) == int else tuple(config['img_size'])
slice_idx = list(range(0, config['img_size'][0], int(config['img_size'][0]/config['display_image_num'])))

for it, (grid, grid_test, image) in enumerate(data_loader):
    # Input coordinates (x, y, z) grid and target image
    matFile = os.path.join(output_directory,'checkdata.mat')
    sio.savemat(matFile,{'grid': grid.numpy(),'grid_test':grid_test.numpy(),'image':image.numpy()})

    grid = grid.cuda()  # [bs, c, h, w, 3], [0, 1]
    grid_test = grid_test.cuda()
    image = image.cuda()  # [bs, c, h, w, 1], [0, 1]
    print('grid size',grid.size())
    print('grid test size',grid_test.size())
    print('image size',image.size())

    # Data loading 
    # Change training inputs for downsampling image
    test_data = (grid_test, image)
    train_data = (grid, image)

    save_image_3d(image[:,:,:,1], slice_idx, os.path.join(image_directory, "train.png"))

    # Train model
    for iterations in range(max_iter):
        model.train()
        optim.zero_grad()

        train_embedding = encoder.embedding(train_data[0])  # [B, C, H, W, embedding*2]
        train_output = model(train_embedding)  # [B, C, H, W, 1]
        #print('train output shape',train_output.size())
        train_loss = 0.5 * loss_fn(train_output, train_data[1])

        train_loss.backward()
        optim.step()

        # Compute training psnr
        if (iterations + 1) % config['log_iter'] == 0:
            train_psnr = -10 * torch.log10(2 * train_loss).item()
            train_loss = train_loss.item()

            #train_writer.add_scalar('train_loss', train_loss, iterations + 1)
            #train_writer.add_scalar('train_psnr', train_psnr, iterations + 1)
            print("[Iteration: {}/{}] Train loss: {:.4g} | Train psnr: {:.4g}".format(iterations + 1, max_iter, train_loss, train_psnr))

        # Compute testing psnr
        if (iterations + 1) % config['val_iter'] == 0:
            model.eval()
            with torch.no_grad():
                test_embedding = encoder.embedding(test_data[0])
                test_output = model(test_embedding)
            save_image_3d(train_output[:,:,:,:,0], slice_idx, os.path.join(image_directory, "recon_{}.png".format(iterations + 1,)))
            save_image_3d(test_output[:,:,:,:,0], slice_idx, os.path.join(image_directory, "recon_iterp_{}.png".format(iterations + 1,)))
            

        if (iterations + 1) % config['image_save_iter'] == 0:
            # Save final model
            model_name = os.path.join(checkpoint_directory, 'model_%06d.pt' % (iterations + 1))
            torch.save({'net': model.state_dict(), \
                        'enc': encoder.B, \
                        'opt': optim.state_dict(), \
                        }, model_name)

   
