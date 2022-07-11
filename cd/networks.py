import numpy as np

import torch
import torch.nn as nn

from utils import check_gpu

############ Input Positional Encoding ############
class Positional_Encoder():
    def __init__(self, args, not_gpu=False):
        if args.encoder['embedding'] == 'gauss':
            self.B = torch.randn((args.encoder['embedding_size'], args.encoder['coordinates_size'])) * args.encoder['scale']
            if not not_gpu:
                self.B = self.B.cuda(args.gpu_id)
        else:
            raise NotImplementedError

    def embedding(self, x):
        #print('x device', x.get_device(), 'b deivce', self.B.get_device())
        x_embedding = torch.matmul((2. * np.pi * x), self.B.t())
        x_embedding = torch.cat([torch.sin(x_embedding), torch.cos(x_embedding)], dim=-1)
        return x_embedding





