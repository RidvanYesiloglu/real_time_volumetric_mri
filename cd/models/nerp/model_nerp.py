import torch
import torch.nn as nn
import numpy as np
import os
from pathlib import Path
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