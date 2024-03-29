import os
#import yaml
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import math
from runset_train import parameters
import glob
import copy
#from data import ImageDataset, ImageDataset_2D, ImageDataset_3D, BeamDataset, BeamDataset_wMask
def find_prev_rec(args):
    if args.im_ind > 1:
        print(f'Args im_ind is now {args.im_ind}')
        prev_args = copy.deepcopy(args)
        prev_args.im_ind = args.im_ind - 1
        print(f'Args im_ind is {args.im_ind} after change of prev_args object.')
        params_dict = parameters.decode_arguments_dictionary('params_dictionary')
        repr_str = parameters.create_repr_str(prev_args, [info.name for info in params_dict.param_infos], wantShort=True, params_dict=params_dict)
        pt_dir = f'{args.main_folder}{prev_args.pt}/'
        prev_res_dir = f'{pt_dir}{prev_args.conf}/t_{prev_args.im_ind}/{repr_str}'
        prev_recs = glob.glob(os.path.join(prev_res_dir, 'rec_*'))
        if ((len(prev_recs) == 0) or (len(prev_recs) > 1)):
            input(f'{len(prev_recs)} prev recs were found! Resolve that and press enter.')
            return find_prev_rec(args)
        rec_path = prev_recs[0]
    elif args.im_ind == 1:
        # use prior_dir as prev_res_dir in this case
        rec_file_name = 'rec_' + args.pri_im_path[:args.pri_im_path.rfind('.')] + '.npy'
        rec_path = f'/home/yesiloglu/projects/real_time_volumetric_mri/priors/{args.pt}/{rec_file_name}'
    else:
        raise ValueError('Invalid im_ind for loading prev rec: {args.im_ind}.')
    print(f'Prev rec was found as: {rec_path}')
    prev_rec = torch.from_numpy(np.load(rec_path)).cuda(args.gpu_id)
    return prev_rec    
# sub=&
def conv_repr_str_to_mlt_line(a_str, sub='&'):
    start = 0
    prev_start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield a_str[prev_start:start]+'\n'
        start += len(sub) # use start += 1 to find overlapping matches
        prev_start = start
        
def PSNR(reconstructed, original): # input, target
    mse = torch.nn.MSELoss()(reconstructed, original)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = original.max()
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr

def check_gpu(gpu_id):
    t = torch.cuda.get_device_properties(gpu_id).total_memory
    r = torch.cuda.memory_reserved(gpu_id)
    a = torch.cuda.memory_allocated(gpu_id)
    f = r-a  # free inside reserved
    print('GPU USAGE: Total: {} MB, Reserved: {} MB, Allocated: {} MB, Free: {} MB'.format(t/1000000.0,r/1000000.0,a/1000000.0,f/1000000.0))

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)

def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory
    

def get_beam_loader(img_path, img_dim_x, img_dim_y,
                    train, batch_size, 
                    num_workers=4, 
                    return_data_idx=False):
    dataset = BeamDataset(img_path,img_dim_x,img_dim_y)
    loader = DataLoader(dataset=dataset, 
                        batch_size=batch_size, 
                        shuffle=train, 
                        drop_last=train, 
                        num_workers=num_workers)
    return loader

def get_beam_loader_wMask(img_path, img_dim_x, img_dim_y, mask_path,
                    train, batch_size, 
                    num_workers=4, 
                    return_data_idx=False):
    dataset = BeamDataset_wMask(img_path,img_dim_x,img_dim_y, mask_path)
    loader = DataLoader(dataset=dataset, 
                        batch_size=batch_size, 
                        shuffle=train, 
                        drop_last=train, 
                        num_workers=num_workers)
    return loader

def get_data_loader(data, img_path, img_dim, img_slice,
                    train, batch_size, 
                    num_workers=4, 
                    return_data_idx=False):
    
    if data == 'phantom':
        dataset = ImageDataset(img_path, img_dim)
    elif '3d' in data:
        dataset = ImageDataset_3D(img_path, img_dim)  # load the whole volume
    else:
        dataset = ImageDataset_2D(img_path, img_dim, img_slice)

    loader = DataLoader(dataset=dataset, 
                        batch_size=batch_size, 
                        shuffle=train, 
                        drop_last=train, 
                        num_workers=num_workers)
    return loader


def save_image_3d(tensor, slice_idx, file_name):
    '''
    tensor: [bs, c, h, w, 1]
    '''
    image_num = len(slice_idx)
    tensor = tensor[0, slice_idx, ...].permute(0, 3, 1, 2).cpu().data  # [c, 1, h, w]
    image_grid = vutils.make_grid(tensor, nrow=image_num, padding=0, normalize=True, scale_each=True)
    vutils.save_image(image_grid, file_name, nrow=1)



def map_coordinates(input, coordinates):
    ''' PyTorch version of scipy.ndimage.interpolation.map_coordinates
    input: (B, H, W, C)
    coordinates: (2, ...)
    '''
    # h = input.shape[0]
    # w = input.shape[1]
    bs, h, w, c = input.size()

    def _coordinates_pad_wrap(h, w, coordinates):
        coordinates[0] = coordinates[0] % h
        coordinates[1] = coordinates[1] % w
        return coordinates

    co_floor = torch.floor(coordinates).long()
    co_ceil = torch.ceil(coordinates).long()
    d1 = (coordinates[1] - co_floor[1].float())
    d2 = (coordinates[0] - co_floor[0].float())
    co_floor = _coordinates_pad_wrap(h, w, co_floor)
    co_ceil = _coordinates_pad_wrap(h, w, co_ceil)

    f00 = input[:, co_floor[0], co_floor[1], :]
    f10 = input[:, co_floor[0], co_ceil[1], :]
    f01 = input[:, co_ceil[0], co_floor[1], :]
    f11 = input[:, co_ceil[0], co_ceil[1], :]
    d1 = d1[None, :, :, None].expand(bs, -1, -1, c)
    d2 = d2[None, :, :, None].expand(bs, -1, -1, c)

    fx1 = f00 + d1 * (f10 - f00)
    fx2 = f01 + d1 * (f11 - f01)
    
    return fx1 + d2 * (fx2 - fx1)


def ct_parallel_project_2d(img, theta):
	bs, h, w, c = img.size()

	# (y, x)=(i, j): [0, w] -> [-0.5, 0.5]
	y, x = torch.meshgrid([torch.arange(h, dtype=torch.float32) / h - 0.5,
							torch.arange(w, dtype=torch.float32) / w - 0.5])

	# Rotation transform matrix: simulate parallel projection rays
	# After rotate xy-axis for theta angle, what is the new (x', y') cooridinates transfromed from original (x, y) cooridnates?
	x_rot = x * torch.cos(theta) - y * torch.sin(theta)
	y_rot = x * torch.sin(theta) + y * torch.cos(theta)

	# Reverse back to index [0, w]
	x_rot = (x_rot + 0.5) * w
	y_rot = (y_rot + 0.5) * h

	# Resampled (x, y) index of the pixel on the projection ray-theta
	sample_coords = torch.stack([y_rot, x_rot], dim=0).cuda()  # [2, h, w]
	# Map the input array to new coordinates by interpolation: spline interpolation order. Boundary extension mode='constant'.
	img_resampled = map_coordinates(img, sample_coords) # [b, h, w, c]

	# Compute integral projections along rays
	proj = torch.mean(img_resampled, dim=1, keepdim=True) # [b, 1, w, c]

	return proj


def ct_parallel_project_2d_batch(img, thetas):
    '''
    img: input tensor [B, H, W, C]
    thetas: list of projection angles
    '''
    projs = []
    for theta in thetas:
    	proj = ct_parallel_project_2d(img, theta)
    	projs.append(proj)
    projs = torch.cat(projs, dim=1)  # [b, num, w, c]

    return projs


def random_sample_gaussian_mask(shape, sample_ratio):
    '''Get binary mask for randomly sampling k-space with normal distribution.
    shape: image or k-space shape (2D or 3D) (N, N, N)
    sample_ratio: downsampling ratio 
    '''
    # Mean and Cov for normal distribtuion centered at the the center of the spectrum
    mean = np.array(shape) // 2  # mean at the center
    cov = np.eye(len(shape)) * (2 * shape[0])  # variance 

    # Sample coordinates (x, y, z) independent to each other with mean = 256 and var = 1024 within range [0, 512]
    nsamp = int(sample_ratio * np.prod(shape))
    samps = np.random.multivariate_normal(mean, cov, size=nsamp).astype(np.int32)  # [N, 3]
    # Within shape index range
    samps = np.clip(samps, 0, shape[0]-1)

    mask = np.zeros(shape)
    inds = []
    for i in range(samps.shape[-1]):
        if i == 2:
            samps[..., i] = ((samps[..., i])//2)
        inds.append(samps[..., i])
    
    mask[tuple(inds)] = 1.
    print('Downsample sparsity:', np.sum(np.abs(mask)) / np.prod(mask.shape))

    return torch.tensor(mask)#.astype(torch.complex64)


def random_sample_uniform_mask(shape, sample_ratio):
    '''Get binary mask for randomly sampling k-space with normal distribution.
    shape: tuple of image shape (2D or 3D) (N, N, N)
    sample_ratio: downsampling ratio 
    '''
    random_array = np.random.rand(*shape)  # uniform distribution over [0, 1)
    mask = random_array < sample_ratio
    mask = mask.astype(np.float32)

    print('Acceleration factor:', 1.0 / sample_ratio)
    print('Downsample sparsity:', np.sum(mask) / np.prod(mask.shape))

    return torch.tensor(mask)


def mri_fourier_transform_2d(image, mask=None):
    '''
    image: input tensor [B, H, W, C]
    mask: mask tensor [H, W]
    '''
    spectrum = torch.fft.fftn(image, dim=(1, 2))

    # K-space spectrum has been shifted to shift the zero-frequency component to the center of the spectrum
    spectrum = torch.fft.fftshift(spectrum, dim=(1, 2))

    # # Downsample k-space
    # spectrum = spectrum * mask[None, :, :, None]

    return spectrum


def mri_fourier_transform_3d(image, mask=None):
    '''
    image: input tensor [B, C, H, W, 1]
    mask: mask tensor [C, H, W]
    '''
    spectrum = torch.fft.fftn(image, dim=(1, 2, 3))

    # K-space spectrum has been shifted to shift the zero-frequency component to the center of the spectrum
    spectrum = torch.fft.fftshift(spectrum, dim=(1, 2, 3))

    # # Downsample k-space
    # spectrum = spectrum * mask[None, :, :, None]

    return spectrum

def complex2real(spectrum):
    real_array = torch.log(torch.abs(spectrum) ** 2)

    return real_array


"""Github code: https://github.com/sifeluga/PDvdi/blob/master/PDvdiSampler.py """
class PoissonSampler2D:
    # those are never changed
    f2PI = math.pi * 2.
    # minimal radius (packing constant) for a fully sampled k-space
    fMinDist = 0.634

    def __init__(self, M=128, N=128, AF=12.0, fPow=2.0, NN=47, aspect=0, tInc=0.):
        self.M = M  # Pattern width
        self.N = N  # Pattern height
        self.fAF = AF  # Acceleration factor (AF) > 1.0
        self.fAspect = aspect
        if aspect == 0:
            self.fAspect = M / N  # Ratio of weight and height (non-square image)
        self.NN = NN  # Number of neighbors (NN) ~[20;80]
        self.fPow = fPow  # Power to raise of distance from center ~[1.5;2.5]
        self.tempInc = tInc  # Temporal incoherence [0;1]

        self.M2 = round(M / 2)  # Center of width
        self.N2 = round(N / 2)  # Center of height
        # need to store density matrix
        self.density = np.zeros((M, N), dtype=np.float32)
        self.targetPoints = round(M * N / AF)

        # init varDens
        if (self.fPow > 0):
            self.variableDensity()

    def variableDensity(self):
        """Precomputes a density matrix, which is used to scale the location-dependent
        radius used for generating new samples.
         """
        fNorm = 1.2 * math.sqrt(pow(self.M2, 2.) + pow(self.N2 * self.fAspect, 2.))

        # computes the euclidean distance for each potential sample location to the center
        for j in range(-self.N2, self.N2, 1):
            for i in range(-self.M2, self.M2, 1):
                self.density[i + self.M2, j + self.N2] = (
                1. - math.sqrt(math.pow(j * self.fAspect, 2.) + math.pow(i, 2.)) / fNorm)

        # avoid diving by zeros
        self.density[(self.density < 0.001)] = 0.001
        # raise scaled distance to the specified power (usually quadratic)
        self.density = np.power(self.density, self.fPow)
        accuDensity = math.floor(np.sum(self.density))

        # linearly adjust accumulated density to match desired number of samples
        if accuDensity != self.targetPoints:
            scale = self.targetPoints / accuDensity
            scale *= 1.0
            self.density *= scale
            self.density[(self.density < 0.001)] = 0.001
            # plt.pcolormesh(self.density)
            # plt.colorbar
            # plt.show()

    def addPoint(self, ptN, fDens, iReg):
        """Inserts a point in the sampling mask if that point is not yet sampled
        and suffices a location-depdent distance (variable density) to
        neighboring points. Returns the index > -1 on success."""
        ptNew = np.around(ptN).astype(np.int, copy=False)
        idx = ptNew[0] + ptNew[1] * self.M

        # point already taken
        if self.mask[ptNew[0], ptNew[1]]:
            return -1

        # check for points in close neighborhood
        for j in range(max(0, ptNew[1] - iReg), min(ptNew[1] + iReg, self.N), 1):
            for i in range(max(0, ptNew[0] - iReg), min(ptNew[0] + iReg, self.M), 1):
                if self.mask[i, j] == True:
                    pt = self.pointArr[self.idx2arr[i + j * self.M]]
                    if pow(pt[0] - ptN[0], 2.) + pow(pt[1] - ptN[1], 2.) < fDens:
                        return -1

        # success if no point was too close
        return idx

    def generate(self, seed, accu_mask=None):

        # set seed for deterministic results
        np.random.seed(seed)

        # preset storage variables
        self.idx2arr = np.zeros((self.M * self.N), dtype=np.int32)
        self.idx2arr.fill(-1)
        self.mask = np.zeros((self.M, self.N), dtype=bool)
        self.mask.fill(False)
        self.pointArr = np.zeros((self.M * self.N, 2), dtype=np.float32)
        activeList = []

        # inits
        count = 0
        pt = np.array([self.M2, self.N2], dtype=np.float32)

        # random jitter of inital point
        jitter = 4
        pt += np.random.uniform(-jitter / 2, jitter / 2, 2)

        # update: point matrix, mask, current idx, idx2matrix and activeList
        self.pointArr[count] = pt
        ptR = np.around(pt).astype(np.int, copy=False)
        idx = ptR[0] + ptR[1] * self.M
        self.mask[ptR[0], ptR[1]] = True
        self.idx2arr[idx] = count
        activeList.append(idx)
        count += 1

        # uniform density
        if (self.fPow == 0):
            self.fMinDist *= self.fAF

        # now sample points
        while (activeList):
            idxp = activeList.pop()
            curPt = self.pointArr[self.idx2arr[idxp]]
            curPtR = np.around(curPt).astype(np.int, copy=False)

            fCurDens = self.fMinDist
            if (self.fPow > 0):
                fCurDens /= self.density[curPtR[0], curPtR[1]]

            region = int(round(fCurDens))

            # if count >= self.targetPoints:
            #    break

            # try to generate NN points around an arbitrary existing point
            for i in range(0, self.NN):
                # random radius and angle
                fRad = np.random.uniform(fCurDens, fCurDens * 2.)
                fAng = np.random.uniform(0., self.f2PI)

                # generate new position
                ptNew = np.array([curPt[0], curPt[1]], dtype=np.float32)
                ptNew[0] += fRad * math.cos(fAng)
                ptNew[1] += fRad * math.sin(fAng)
                ptNewR = np.around(ptNew).astype(np.int, copy=False)
                # continue when old and new positions are the same after rounding
                if ptNewR[0] == curPtR[0] and ptNewR[1] == curPtR[1]:
                    continue

                if (ptNewR[0] >= 0 and ptNewR[1] >= 0 and ptNewR[0] < self.M and ptNewR[1] < self.N):
                    newCurDens = self.fMinDist / self.density[ptNewR[0], ptNewR[1]]
                    if self.fPow == 0:
                        newCurDens = self.fMinDist
                    if self.tempInc > 0 and accu_mask is not None:
                        if accu_mask[ptNewR[0], ptNewR[1]] > self.density[
                            ptNewR[0], ptNewR[1]] * seed + 1.01 - self.tempInc:
                            continue
                    idx = self.addPoint(ptNew, newCurDens, region)
                    if idx >= 0:
                        self.mask[ptNewR[0], ptNewR[1]] = True
                        self.pointArr[count] = ptNew
                        self.idx2arr[idx] = count
                        activeList.append(idx)
                        count += 1

        print("Generating finished with " + str(count) + " points.")

        return torch.tensor(self.mask.astype(np.float32)).to(torch.complex64)


# def write_samples(str,mask):
#     f = open(str,'w')
#     for i in range(0,mask.shape[0]):
#         for j in range(0, mask.shape[1]):
#             if mask[i,j] > 0:
#                 f.write('%d %d\n' % (i, j))
#     f.close()



