import torch
import torch.nn as nn
import numpy as np
import os
from pathlib import Path
from torchnufftexample import create_radial_mask, project_radial, backproject_radial
from skimage.metrics import structural_similarity as ssim
from utils import NoamOpt
from jacobian import JacobianReg

class Main_Module(nn.Module):
    def __init__(self, args):
        super().__init__()
        print('reg is ', args.reg)
        if sum([args.conf==item for item in ['pri_emb','trn_wo_trns','trn_w_trns']])  == 0:
            raise ValueError('Invalid configuration')
        if (args.conf == 'pri_emb') and (args.reg is not None):
            raise ValueError(f'In prior embedding, no regularization can be used but {args.reg} is specified.')
        self.conf = args.conf
        self.optims = []
        grid_np = np.asarray([(x,y,z) for x in range(128) for y in range(128) for z in range(64)]).reshape((1,128,128,64,3))
        grid_np = (grid_np/np.array([128,128,64])) + np.array([1/256.0,1/256.0,1/128.0])
        grid = torch.from_numpy(grid_np.astype('float32')).reshape(-1,3)
        grid = grid.cuda(args.gpu_id)
        self.grid = grid
        self.grid.requires_grad = True
        self.mse_loss_fn = torch.nn.MSELoss()
        self.im_shape = (1,128,128,64,1)
        
        im_dir = args.data_dir + args.pt + '/all_vols.npy'
        self.image = torch.from_numpy(np.expand_dims(np.load(im_dir)[args.im_ind],(0,-1)).astype('float32')).cuda(args.gpu_id)
        print('iamge dshpae', self.image.shape)
        self.im_nerp_enc = Positional_Encoder(args)
        self.im_nerp_mlp = SIREN(args.net_inp_sz, args.net_wd, args.net_dp, args.net_ou_sz)
        self.im_nerp_mlp.cuda(args.gpu_id)
        self.im_nerp_mlp.train()
        if args.ld_pri_im:
            args.we_dec_co=0
        optim_im_nerp_mlp = torch.optim.Adam(self.im_nerp_mlp.parameters(), lr=args.lr_im, betas=(args.beta1, args.beta2), weight_decay=args.we_dec_co)
        self.optims.append(optim_im_nerp_mlp)
        if args.ld_pri_im:
            prior_dir = f'/home/yesiloglu/projects/real_time_volumetric_mri/priors/{args.pt}/'
            state_dict = torch.load(prior_dir+args.pri_im_path, map_location=lambda storage, loc: storage.cuda(args.gpu_id))
            self.im_nerp_mlp.load_state_dict(state_dict['im_nerp_mlp'])
            self.im_nerp_enc.B = state_dict['im_nerp_enc']#.cuda(args.gpu_id)
            self.im_nerp_mlp = self.im_nerp_mlp.cuda(args.gpu_id)
            # for no,optim in enumerate(self.optims):
            #     optim.load_state_dict(state_dict[f'opt{no}'].state_dict())
            print('Load prior model: {}'.format(prior_dir+args.pri_im_path))
        if self.conf != 'pri_emb': #gt_kdata and ktraj needed for loss calc
            self.ktraj, self.im_size_for_rad, self.grid_size_for_rad = create_radial_mask(args.nproj, (64,1,128,128), args.gpu_id, plot=False)
            self.gt_kdata = project_radial(self.image, self.ktraj, self.im_size_for_rad, self.grid_size_for_rad)
        if self.conf == 'trn_w_trns':
            self.tr_nerp_enc = Positional_Encoder(args)
            self.tr_nerp_mlp = SIREN(args.tr_inp_sz, args.tr_wd, args.tr_dp, args.tr_ou_sz)
            self.tr_nerp_mlp.cuda(args.gpu_id)
            self.tr_nerp_mlp.train()
            # Setup optimizer for transformation nerp:
            optim_tr_nerp_mlp = torch.optim.Adam(self.tr_nerp_mlp.parameters(), lr=args.lr_tr, betas=(args.beta1, args.beta2), weight_decay=args.we_dec_co)
            self.optims.append(optim_tr_nerp_mlp)
            if args.use_jc_grid_reg:
                self.jacob_reg = JacobianReg(gpu_id=args.gpu_id)
                self.lambda_JR = args.lambda_JR
                self.tr_nerp_mlp = nn.DataParallel(self.tr_nerp_mlp,[args.gpu_id,args.gpu2_id])
                self.im_nerp_mlp = nn.DataParallel(self.im_nerp_mlp,[args.gpu_id, args.gpu2_id])
    def forward(self):
        if self.conf == 'pri_emb':
            output_im = self.im_nerp_mlp(self.im_nerp_enc.embedding(self.grid))
            output_im = output_im.reshape(self.im_shape)
            train_loss = self.mse_loss_fn(output_im, self.image)
        elif self.conf == 'trn_wo_trns':
            output_im = self.im_nerp_mlp(self.im_nerp_enc.embedding(self.grid))
            output_im = output_im.reshape(self.im_shape)
            #train_spec = mri_fourier_transform_3d(output_im)  # [B, H, W, C]
            #train_spec = train_spec * preruni_dict['mask'][None, ..., None]
            out_kspace = project_radial(output_im, self.ktraj, self.im_size_for_rad, self.grid_size_for_rad)
            train_loss = self.mse_loss_fn(out_kspace, self.gt_kdata)
        elif self.conf == 'trn_w_trns':
            deformed_grid = self.grid + (self.tr_nerp_mlp(self.tr_nerp_enc.embedding(self.grid)))  # [B, C, H, W, 1]
            output_im = self.im_nerp_mlp(self.im_nerp_enc.embedding(deformed_grid))
            output_im = output_im.reshape(self.im_shape)
            out_kspace = project_radial(output_im, self.ktraj, self.im_size_for_rad, self.grid_size_for_rad)
            if self.jacob_reg is not None:
                
                grid_reg_loss = self.jacob_reg(self.grid, deformed_grid)   # Jacobian regularization
                train_loss = self.mse_loss_fn(out_kspace, self.gt_kdata) + self.lambda_JR*grid_reg_loss
            else:
                train_loss = self.mse_loss_fn(out_kspace, self.gt_kdata)
        return train_loss
    
    def test_psnr_ssim(self, ret_im=False):
        with torch.no_grad():
            if self.conf == 'pri_emb':
                output_im = self.im_nerp_mlp(self.im_nerp_enc.embedding(self.grid))
                output_im = output_im.reshape(self.im_shape)
                test_loss = self.mse_loss_fn(output_im, self.image).item()
            elif self.conf == 'trn_wo_trns':
                output_im = self.im_nerp_mlp(self.im_nerp_enc.embedding(self.grid))
                output_im = output_im.reshape(self.im_shape)
                test_kdata = project_radial(output_im, self.ktraj, self.im_size_for_rad, self.grid_size_for_rad)
                test_loss = self.mse_loss_fn(test_kdata, self.gt_kdata).item()
            elif self.conf == 'trn_w_trns':
                deformed_grid = self.grid + (self.tr_nerp_mlp(self.tr_nerp_enc.embedding(self.grid)))  # [B, C, H, W, 1]
                output_im = self.im_nerp_mlp(self.im_nerp_enc.embedding(deformed_grid))
                output_im = output_im.reshape(self.im_shape)
                out_kspace = project_radial(output_im, self.ktraj, self.im_size_for_rad, self.grid_size_for_rad)
                test_loss = self.mse_loss_fn(out_kspace, self.gt_kdata)
        test_psnr = 20*torch.log10(self.image.max()).item() - 10 * torch.log10(self.mse_loss_fn(output_im, self.image)).item()
        test_ssim = ssim(output_im.cpu().numpy().squeeze(), self.image.cpu().numpy().squeeze(), data_range=self.image.max().item()-self.image.min().item())
        if ret_im:
            return output_im, test_psnr , test_ssim, test_loss
        else:
            return test_psnr , test_ssim, test_loss

    def get_to_save_dict(self):
        to_save_dict = {'im_nerp_mlp': self.im_nerp_mlp.state_dict(), \
                    'im_nerp_enc':self.im_nerp_enc.B}
        if self.conf == 'trn_w_trns':
            to_save_dict['tr_nerp_mlp'] = self.tr_nerp_mlp
            to_save_dict['tr_nerp_enc'] = self.tr_nerp_enc
        for no,optim in enumerate(self.optims):
            to_save_dict[f'opt{no}'] = self.optims[no].state_dict()
        return to_save_dict
############ Input Positional Encoding ############
class Positional_Encoder():
    def __init__(self, args, not_gpu=False):
        if args.enc_emb == 'gauss':
            self.B = torch.randn((args.enc_emb_sz, args.enc_crd_sz)) * args.enc_scale
            if not not_gpu:
                self.B = self.B.cuda(args.gpu_id)
        else:
            raise NotImplementedError

    def embedding(self, x):
        #print('x device', x.get_device(), 'b deivce', self.B.get_device())
        x_embedding = torch.matmul((2. * np.pi * x), self.B.t())
        x_embedding = torch.cat([torch.sin(x_embedding), torch.cos(x_embedding)], dim=-1)
        return x_embedding

############ Fourier Feature Network ############
class Swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):      
        return x * torch.sigmoid(x)

class FFN(nn.Module):
    def __init__(self, params):
        super(FFN, self).__init__()

        num_layers = params['network_depth']
        hidden_dim = params['network_width']
        input_dim = params['network_input_size']
        output_dim = params['network_output_size']

        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for i in range(1, num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out



############ Fourier Feature Network ############
class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()

    def init_weights(self):
        b = 1 / \
            self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        # print('Sirenlayer incoming x.shape: {}: '.format(x.shape))
        # check_gpu(1)
        x = self.linear(x)
        # print('Sirenlayer x.shape after linear: {}: '.format(x.shape))
        # check_gpu(1)
        return x if self.is_last else torch.sin(self.w0 * x)


class SIREN(nn.Module):
    def __init__(self, net_inp_sz, net_wd, net_dp, net_ou_sz):
        super(SIREN, self).__init__()

        input_dim = net_inp_sz
        hidden_dim = net_wd
        num_layers = net_dp
        output_dim = net_ou_sz

        layers = [SirenLayer(input_dim, hidden_dim, is_first=True)]
        for i in range(1, num_layers - 1):
            layers.append(SirenLayer(hidden_dim, hidden_dim))
        layers.append(SirenLayer(hidden_dim, output_dim, is_last=True))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # print('SIREN forwardi. x.shape: {}'.format(x.shape))
        # check_gpu(1)
        out = self.model(x)

        return out