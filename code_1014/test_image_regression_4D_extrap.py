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

cudnn.benchmark = True
config['img_size'] = np.array([config['z_dim'],config['y_dim'],config['x_dim']])
config['img_idArray'] = np.array([config['img_id1'],config['img_id2']])
print('img size info',np.shape(config['img_size']))

# Setup output folder
output_folder = os.path.splitext(os.path.basename(opts.config))[0]
model_name = 'img{}_{}_{}_{}_{}_{}_lr{:.2g}_encoder_{}' \
    .format(config['img_size'], config['model'], \
        config['net']['network_input_size'], config['net']['network_width'], \
        config['net']['network_depth'], config['loss'], config['lr'], config['encoder']['embedding'])

if not(config['encoder']['embedding'] == 'none'):
    model_name += '_scale{}_size{}'.format(config['encoder']['scale'], config['encoder']['embedding_size'])


output_directory = os.path.join(opts.output_path + "/outputs", output_folder, config['data'], model_name)
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

pretrain_path = config['pretrain_path']
pretrain_model = os.path.join(pretrain_path,model_name,'checkpoints','model_001000.pt')
print(pretrain_model)
state_dict = torch.load(pretrain_model)
model.load_state_dict(state_dict['net'])
encoder.B = state_dict['enc']
model.eval()

increment = 0.01/10 # create 10 phases

pos_end = config['pos_end']
neg_end = config['neg_end']
for it in range(pos_end):
    endpoint = (it+1)*increment
    grid_z, grid_y, grid_x, grid_t = torch.meshgrid([torch.linspace(0, 1, steps=config['z_dim']), \
                                            torch.linspace(0, 1, steps=config['y_dim']), \
                                            torch.linspace(0, 1, steps=config['x_dim']),\
                                            torch.linspace(0,endpoint,steps=len(config['img_idArray']))])
    grid = torch.stack([grid_z, grid_y, grid_x, grid_t], dim=-1)   
    grid = grid.cuda()  # [bs, c, h, w, 3], [0, 1]
    test_embedding = encoder.embedding(grid)
    test_output = model(test_embedding)

    matFile = os.path.join(output_directory,'interp_'+str(it+1)+'.mat')
    sio.savemat(matFile,{'grid': grid_t.numpy(),'iterp':test_output.detach().cpu().numpy()})


for it in range(neg_end):

    endpoint = (it+1)*increment*(-1)
    grid_z, grid_y, grid_x, grid_t = torch.meshgrid([torch.linspace(0, 1, steps=config['z_dim']), \
                                            torch.linspace(0, 1, steps=config['y_dim']), \
                                            torch.linspace(0, 1, steps=config['x_dim']),\
                                            torch.linspace(endpoint,0,steps=len(config['img_idArray']))])
    grid = torch.stack([grid_z, grid_y, grid_x, grid_t], dim=-1)   
    grid = grid.cuda()  # [bs, c, h, w, 3], [0, 1]
    test_embedding = encoder.embedding(grid)
    test_output = model(test_embedding)

    matFile = os.path.join(output_directory,'interp_'+str((it+1)*(-1))+'.mat')
    sio.savemat(matFile,{'grid': grid_t.numpy(),'iterp':test_output.detach().cpu().numpy()})

  
