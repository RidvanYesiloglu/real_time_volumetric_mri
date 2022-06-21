# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 17:13:47 2021

@author: ridva
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from mpl_toolkits.axes_grid1 import make_axes_locatable

def normalize(img, min_val=None, max_val=None):
    min_val = min_val if min_val is not None else img.min()
    max_val = max_val if max_val is not None else img.max()
    return (img-min_val)/(max_val-min_val)
def extract_vol_no(name):
    return int(name[name.find('_')+1:name.find('.')])
def psnr_arr(ims):
    return -10*np.log(((ims[0:1]-ims[1:])**2).mean((1,2,3)))
dtname = 'pt_95_5min'
data_folder = os.path.join('data_4DMRI',dtname) # 'patient19'
patient_list = os.listdir(data_folder)
patient_list.sort(key=extract_vol_no)

# if data_folder == 'data_4DMRI':
#     patient_list = os.listdir(data_folder)
#     patient_list.sort(key=extract_vol_no)
# elif data_folder == 'patient19':
#     with open(os.path.join(data_folder, 'MLP_List.txt'), 'r') as patient_list_file:
#         patient_list = patient_list_file.read().splitlines()
    
arr = np.zeros((len(patient_list),128,128))
maxs = np.zeros((len(patient_list)))
mins = np.zeros((len(patient_list)))
ims_sep = np.zeros((len(patient_list),128,128,64))
ims_tog = np.zeros((len(patient_list),128,128,64))
for no,pat in enumerate(patient_list):
    img = np.fromfile(os.path.join(data_folder,pat), dtype='float32')
    maxs[no]=img.max()
    mins[no]=img.min()
    reshape_size = (128,128,64)
    #if no==44 or no==57:
    ims_tog[no]=(np.reshape(img.transpose(), reshape_size, order="F"))#ax[int(no>50)].imshow(img[...,0])
    ims_sep[no]=normalize(np.reshape(img.transpose(), reshape_size, order="F"))#ax[int(no>50)].imshow(img[...,0])
    #img = normalize(np.reshape(img.transpose(), reshape_size, order="F"))

    #np.save(f'3d_tracking_nerp/data73/{pat[:-4]}',img)
#    arr[no] = img[:,:,15]
   # plt.imshow(img[:,:,15],cmap='gray', interpolation='none')
   # print(no, img.shape)
ims_tog = normalize(ims_tog)
ims_tog = ims_tog[np.concatenate((np.array([True]), psnr_arr(ims_tog)!=np.inf))]
ims_sep = ims_sep[np.concatenate((np.array([True]), psnr_arr(ims_sep)!=np.inf))]
np.save(os.path.join('dt_4DMRI', dtname, 'ims_tog'),ims_tog)
np.save(os.path.join('dt_4DMRI', dtname, 'ims_sep'),ims_sep)
plot=False
if plot:
    fig,ax = plt.subplots(2,2)
    im00 = ax[0,0].imshow(ims_tog[0,...,30],cmap='gray')#, vmin=ims_tog.min(), vmax=ims_tog.max())
    im01 = ax[0,1].imshow(ims_tog[70,...,30],cmap='gray')#, vmin=ims_tog.min(), vmax=ims_tog.max())
    im10 = ax[1,0].imshow(ims_sep[0,...,25],cmap='gray', vmin=ims_sep.min(), vmax=ims_sep.max())
    im11 = ax[1,1].imshow(ims_sep[30,...,25],cmap='gray', vmin=ims_sep.min(), vmax=ims_sep.max())
    # ax[0,0].axis('off')
    # ax[0,1].axis('off')
    # ax[1,0].axis('off')
    # ax[1,1].axis('off')
    ax[0,0].set_title('44th Time Point, 63rd Slice')
    ax[0,1].set_title('57th Time Point, 63rd Slice')
    divider = make_axes_locatable(ax[0,0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im00, cax=cax, orientation='vertical')
    divider = make_axes_locatable(ax[0,1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im01, cax=cax, orientation='vertical')
    divider = make_axes_locatable(ax[1,0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im10, cax=cax, orientation='vertical')
    divider = make_axes_locatable(ax[1,1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im11, cax=cax, orientation='vertical')
    pad=-1
    ax[0,0].annotate('Normalized Together', xy=(0, 0.5), xytext=(-ax[0,0].yaxis.labelpad - pad, 0),
                    xycoords=ax[0,0].yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')
    ax[1,0].annotate('Normalized Seperately', xy=(0, 0.5), xytext=(-ax[1,0].yaxis.labelpad - pad, 0),
                    xycoords=ax[1,0].yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')
    fig.tight_layout()
    plt.show()
    
    fig,ax = plt.subplots()
    ax.plot(mins, label='mins')
    ax.plot(maxs, label='maxs')
    ax.set_xlabel('Time Point')
    ax.set_ylabel('Min and Max Value')
    plt.legend()
    plt.show()
    
    bg_color = '#95A4AD'
    bar_color = '#283F4E'
    gif_name = '15ler'
    
    filenames = []
    for i in np.arange(0, arr.shape[0]):
        filename = f'deneme/frame_{i}.png'
        filenames.append(filename)
        
        # last frame of each viz stays longer
        if (i == (arr.shape[0]-1)):
            for i in range(5):
                filenames.append(filename)
                
        # save img
        plt.imshow(arr[i,:,:], cmap='gray', interpolation='none')
        plt.title(f"Image {i}")
        plt.savefig(filename, dpi=96, bbox_inches='tight')
        plt.close()
    print('Charts saved\n')
    # Build GIF
    print('Creating gif\n')
    with imageio.get_writer(f'{gif_name}.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    print('Gif saved\n')
    print('Removing Images\n')
    # Remove files
    for filename in set(filenames):
        os.remove(filename)
    print('DONE')

def findxyz(a):
    x = a//(128*64)
    remx = a - x*128*64
    y = remx // 64
    remy = remx - y*64
    z = remy
    return x,y,z