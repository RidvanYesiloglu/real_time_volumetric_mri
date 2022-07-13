# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 13:49:46 2022

@author: ridva
"""
import matplotlib.pyplot as plt
import numpy as np
p_p = np.load('psnr_p_pt19.npy')
p = np.load('psnr_pt19.npy')
s = np.load('ssim_ldprvnt_pt19.npy')
s_p = np.load('ssim_p_pt19.npy')


fig,ax = plt.subplots(2,1)
#ax.plot(p_p, label='Prior psnr')
ax[0].plot(p[:19])
ax[0].set_title('PSNR of reconstruction with init. from previous image NeRP')
ax[1].plot(s[:19])
ax[1].set_title('SSIM of reconstruction with init. from previous image NeRP')
plt.tight_layout()
#plt.legend()
