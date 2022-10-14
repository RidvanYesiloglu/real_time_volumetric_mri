import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def display_arr_stats(arr):
    shape, vmin, vmax, vmean, vstd = arr.shape, np.min(arr), np.max(arr), np.mean(arr), np.std(arr)
    print('shape:{} | min:{:.3f} | max:{:.3f} | mean:{:.3f} | std:{:.3f}'.format(shape, vmin, vmax, vmean, vstd))


def display_tensor_stats(tensor):
    shape, vmin, vmax, vmean, vstd = tensor.shape, tensor.min(), tensor.max(), torch.mean(tensor), torch.std(tensor)
    print('shape:{} | min:{:.3f} | max:{:.3f} | mean:{:.3f} | std:{:.3f}'.format(shape, vmin, vmax, vmean, vstd))


def create_grid(h, w):
    grid_y, grid_x = torch.meshgrid([torch.linspace(0, 1, steps=h),
                                     torch.linspace(0, 1, steps=w)])
    grid = torch.stack([grid_y, grid_x], dim=-1)
    return grid


def create_grid_3d(c, h, w):
    grid_z, grid_y, grid_x = torch.meshgrid([torch.linspace(0, 1, steps=c), \
                                            torch.linspace(0, 1, steps=h), \
                                            torch.linspace(0, 1, steps=w)])
    grid = torch.stack([grid_z, grid_y, grid_x], dim=-1)
    return grid

def create_grid_4d(c, h, w, t):
    print('c ',c,' h ',h,' w ',w, ' t ',t)
    grid_z, grid_y, grid_x, grid_t = torch.meshgrid([torch.linspace(0, 1, steps=c), \
                                            torch.linspace(0, 1, steps=h), \
                                            torch.linspace(0, 1, steps=w),\
                                            torch.linspace(0,0.01,steps=t)])
    grid = torch.stack([grid_z, grid_y, grid_x, grid_t], dim=-1)
    grid_test = torch.stack([grid_z, grid_y, grid_x, grid_t+0.01/2], dim=-1)
    return grid, grid_test
    
def create_grid_3d_nouniform(c, h, w):
    nF = max(c,h,w)
    r1 = c/nF
    r2 = h/nF
    r3 = w/nF
    grid_z, grid_y, grid_x = torch.meshgrid([torch.linspace(0, r1, steps=c), \
                                            torch.linspace(0, r2, steps=h), \
                                            torch.linspace(0, r3, steps=w)])
    grid = torch.stack([grid_z, grid_y, grid_x], dim=-1)
    return grid    
    
class ImageDataset_2D_multiview(Dataset):

    def __init__(self, img_path, img_dim):
        '''
        img_dim: new image size [z, h, w]
        '''
        self.img_dim = (img_dim, img_dim, img_dim) if type(img_dim) == int else tuple(img_dim)
        image = np.fromfile(img_path,'float32')  # [C, H, W]
        image = np.reshape(image,img_dim)
        image = torch.tensor(image, dtype=torch.float32)[None, ...]  # [B, C, H, W]

        # Scaling normalization
        image = image / torch.max(image)  # [B, C, H, W], [0, 1]
        self.img = image.permute(1, 2, 3, 0)  # [C, H, W, 1]
        display_tensor_stats(self.img)

    def __getitem__(self, idx):
        grid = create_grid_3d_nouniform(*self.img_dim)
        return grid, self.img

    def __len__(self):
        return 1

class ImageDataset_3D(Dataset):

    def __init__(self, img_path, img_dim):
        '''
        img_dim: new image size [z, h, w]
        '''
        self.img_dim = (img_dim, img_dim, img_dim) if type(img_dim) == int else tuple(img_dim)
        image = np.fromfile(img_path,'float32')  # [C, H, W]
        image = np.reshape(image,img_dim)

        # Crop slices in z dim
        center_idx = int(image.shape[0] / 2)
        num_slice = int(self.img_dim[0] / 2)
        image = image[center_idx-num_slice:center_idx+num_slice, :, :]
        im_size = image.shape
        print(image.shape, center_idx, num_slice)

        # Complete 3D input image as a squared x-y image
        if not(im_size[1] == im_size[2]):
            zerp_padding = np.zeros([im_size[0], im_size[1], np.int((im_size[1]-im_size[2])/2)])
            image = np.concatenate([zerp_padding, image, zerp_padding], axis=-1)

        # Resize image in x-y plane
        image = torch.tensor(image, dtype=torch.float32)[None, ...]  # [B, C, H, W]
        image = F.interpolate(image, size=(self.img_dim[1], self.img_dim[2]), mode='bilinear', align_corners=False)

        # Scaling normalization
        image = image / torch.max(image)  # [B, C, H, W], [0, 1]
        self.img = image.permute(1, 2, 3, 0)  # [C, H, W, 1]
        display_tensor_stats(self.img)

    def __getitem__(self, idx):
        grid = create_grid_3d(*self.img_dim)
        return grid, self.img

    def __len__(self):
        return 1

class ImageDataset_4D(Dataset):

    def __init__(self, img_path, img_dim, img_idArray):
        '''
        img_dim: new image size [z, h, w]

        '''

        #self.img_dim = (img_dim, img_dim, img_dim) if type(img_dim) == int else tuple(img_dim)
        t_dim = len(img_idArray)
        data_dim = [img_dim[0],img_dim[1],img_dim[2],t_dim]
        self.data_dim = data_dim
        img = torch.zeros(data_dim)
        
        counter = 0
        for img_idx in img_idArray:
            image = np.fromfile(img_path.format(img_idx),'float32')  # [C, H, W]
            image = np.reshape(image,img_dim)          
            image = torch.tensor(image, dtype=torch.float32)  # [B, C, H, W] 
            # Scaling normalization
            print('one image shape',image.size())
            image = image / torch.max(image)  # [B, C, H, W], [0, 1]
            img[:,:,:,counter] = image  # [C, H, W, 1]
            counter = counter+1
        img = img.unsqueeze(0)
        self.img = img.permute([1,2,3,4,0])

        display_tensor_stats(self.img)

    def __getitem__(self, idx):
        grid, grid_test = create_grid_4d(*self.data_dim)
        return grid, grid_test, self.img

    def __len__(self):
        return 1


class ImageDataset_2D(Dataset):

    def __init__(self, img_path, img_dim, img_slice):
        '''
        img_dim: new image size [h, w]
        '''
        self.img_dim = (img_dim, img_dim) if type(img_dim) == int else tuple(img_dim)
        image = np.load(img_path)['data']  # [100, 320, 260] (0, 1)
        image = image[img_slice, :, :]  # Choose one slice as 2D CT image
        imsize = image.shape

        # Complete as a squared image
        if not(imsize[0] == imsize[1]):
            zerp_padding = np.zeros([imsize[0], np.int((imsize[0] - imsize[1])/2)])
            image = np.concatenate([zerp_padding, image, zerp_padding], axis=1)

        # Definition of [H, W] for cv2.resize is reversed [W, H]
        image = cv2.resize(image, self.img_dim[::-1], interpolation=cv2.INTER_LINEAR)  # Interpolate image to predefined size

        # Scaling normalization
        image = image / np.max(image)  # [0, 1]
        self.img = torch.tensor(image, dtype=torch.float32)[:, :, None]
        display_tensor_stats(self.img)

        
    def __getitem__(self, idx):
        
        # grid = create_grid(*self.img_dim[::-1])
        grid = create_grid(*self.img_dim)
        return grid, self.img

    def __len__(self):
        return 1


class ImageDataset(Dataset):

    def __init__(self, img_path, img_dim):
        self.img_dim = (img_dim, img_dim) if type(img_dim) == int else img_dim

        # image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  
        h, w = image.shape
        left_w = int((w - h) / 2)

        image = image[:, left_w:left_w + h]
        image = cv2.resize(image, self.img_dim, interpolation=cv2.INTER_LINEAR)
        self.img = image

    def __getitem__(self, idx):
        image = self.img / 255  # [0, 1]
        # display_arr_stats(image)
        grid = create_grid(*self.img_dim[::-1])  # [0, 1]

        return grid, torch.tensor(image, dtype=torch.float32)[:, :, None]

    def __len__(self):
        return 1


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import torchvision

#     ds = ImageDataset("data/fox.jpg", 512)
#     grid, image = ds[0]
#     torchvision.utils.save_image(image.permute(2, 0, 1), "data/demo.jpg")
#     image = image.numpy() * 255
#     plt.imshow(image.astype(np.uint8))
#     plt.show()
