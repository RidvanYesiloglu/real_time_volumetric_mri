import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

class data_linewidth_plot():
    def __init__(self, x, y, **kwargs):
        self.ax = kwargs.pop("ax", plt.gca())
        self.fig = self.ax.get_figure()
        self.lw_data = kwargs.pop("linewidth", 1)
        self.lw = 1
        self.fig.canvas.draw()

        self.ppd = 72./self.fig.dpi
        self.trans = self.ax.transData.transform
        self.linehandle, = self.ax.plot([],[],**kwargs)
        if "label" in kwargs: kwargs.pop("label")
        self.line, = self.ax.plot(x, y, **kwargs)
        self.line.set_color(self.linehandle.get_color())
        self._resize()
        self.cid = self.fig.canvas.mpl_connect('draw_event', self._resize)

    def _resize(self, event=None):
        lw =  ((self.trans((1, self.lw_data))-self.trans((0, 0)))*self.ppd)[1]
        if lw != self.lw:
            self.line.set_linewidth(lw)
            self.lw = lw
            self._redraw_later()

    def _redraw_later(self):
        self.timer = self.fig.canvas.new_timer(interval=10)
        self.timer.single_shot = True
        self.timer.add_callback(lambda : self.fig.canvas.draw_idle())
        self.timer.start()
        
def calc_3dpsnrs_wrti(all_vols):
    return 20*np.log((all_vols[0:1]**2).max((1,2,3)))-10*np.log(((all_vols[0:1]-all_vols)**2).mean((1,2,3)))
def calc_psnrs_wrti(all_vols, ax_cr_sg):
    if ax_cr_sg == 0:
        return 20*np.log((all_vols[0:1]**2).max((1,2)))-10*np.log(((all_vols[0:1]-all_vols)**2).mean((1,2)))
    elif ax_cr_sg == 1:
        return 20*np.log((all_vols[0:1]**2).max((1,3)))-10*np.log(((all_vols[0:1]-all_vols)**2).mean((1,3)))
    elif ax_cr_sg == 2:
        return 20*np.log((all_vols[0:1]**2).max((2,3)))-10*np.log(((all_vols[0:1]-all_vols)**2).mean((2,3)))
def main(all_vols, pt_id):
    # Calculate 3d PSNRs with respect to the initial image
    psnrs = calc_3dpsnrs_wrti(all_vols)
    print(psnrs[547], ' db 3dpsnr')
    # Create and save the plot of PSNRs:
    fig,ax = plt.subplots(figsize=(15,6))
    ax2 = ax.twinx()
    ax.plot(psnrs,'.-', color='black')
    ax2.scatter(np.where(psnrs==np.inf)[0], np.ones(((psnrs==np.inf).sum())), marker='.', color='black')
    ax2.set_ylim([0,1])
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    ax.set_title('3D PSNRs wrt the Initial Image')
    ax.set_xlabel('Time Index')
    ax.set_ylabel('PSNR (dB)')
    ax.set_xticks(np.arange(0,psnrs.shape[0],step=50))
    ax.set_yticks(np.concatenate((ax.get_yticks(),ax.get_yticks()[-1:]+20)))
    ax.set_yticklabels(np.concatenate(([str(int(tick)) for tick in ax.get_yticks()[:-1]],np.array(['inf']))))
    plt.show()
    plt.savefig(f'/raid/yesiloglu/data/real_time_volumetric_mri/{pt_id}/temporal_evol_gifs/3dpsnrs_wrt_init_vs_t_{pt_id}', dpi=96, bbox_inches='tight')
    plt.close(fig)
    
    psnrs = np.concatenate([calc_psnrs_wrti(all_vols, ax_cr_sg) for ax_cr_sg in [0,1,2]], axis=1).T
    #psnrs = np.random.rand(320,876)
    sep = 10
    sl_no_mplier = 1
    psnrs_sep = np.concatenate((psnrs[:all_vols.shape[3], :], np.ones((sep,psnrs.shape[1]))*psnrs.min(), psnrs[all_vols.shape[3]:all_vols.shape[2]+all_vols.shape[3], :], np.ones((sep,psnrs.shape[1]))*psnrs.min(), psnrs[all_vols.shape[2]+all_vols.shape[3]:, :]),0)
    
    fig,ax=plt.subplots(figsize=(19,10))
    
    shown_im = ax.imshow(np.repeat(psnrs_sep, sl_no_mplier, axis=0))
    ax.set_title('Axial, Coronal and Sagittal PSNRs wrt the Initial Image', size='x-large')
    xticklabels = np.arange(0,psnrs.shape[1],step=30)
    ax.set_xticklabels(xticklabels)
    ax.set_xticks(xticklabels)
    
    yticks = sl_no_mplier*np.arange(0,psnrs.shape[0],step=8)
    
    yticklabels = yticks.copy()//sl_no_mplier
    is_ax = (yticklabels<all_vols.shape[3]).astype(bool)
    is_cr = ((1 - is_ax) & (yticklabels<(all_vols.shape[2] + all_vols.shape[3]))).astype(bool)
    is_sg = ((1 - is_ax) & (1 - is_cr)).astype(bool)
    yticklabels[is_cr] = yticklabels[is_cr] - all_vols.shape[3]
    yticklabels[is_sg] = yticklabels[is_sg] - all_vols.shape[2] - all_vols.shape[3]
    
    
    yticks[is_cr] = yticks[is_cr] + sep*sl_no_mplier
    yticks[is_sg] = yticks[is_sg] + 2*sep*sl_no_mplier
    
    ax.set_yticks(yticks + sl_no_mplier//2)
    ax.set_yticklabels(yticklabels)
    
    ax.text(-22,sl_no_mplier*(all_vols.shape[3]/2)+sl_no_mplier//2,'Axial Slices', va='center', rotation='vertical')
    ax.text(-22,sl_no_mplier*(all_vols.shape[3] + sep + all_vols.shape[2]/2) + sl_no_mplier//2,'Coronal Slices', va='center', rotation='vertical')
    ax.text(-22,sl_no_mplier*(all_vols.shape[3] + sep + all_vols.shape[2] + sep + all_vols.shape[1]/2) + sl_no_mplier//2,'Sagittal Slices', va='center', rotation='vertical')
    ax.text(-30, sl_no_mplier*(psnrs_sep.shape[0]/2) + sl_no_mplier//2, 'Slice No', va='center', ha='center', rotation='vertical', size='large')
    ax.text(psnrs_sep.shape[1]/2, sl_no_mplier*(psnrs_sep.shape[0]) + sl_no_mplier//2 + 18, 'Time Index', va='center', ha='center', rotation='horizontal', size='large')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.20)
    fig.colorbar(shown_im, cax=cax, orientation='vertical')
    sep_loc_1 = sl_no_mplier*(all_vols.shape[3]+sep/2)+sl_no_mplier//2-0.5
    data_linewidth_plot([0.5, psnrs.shape[1]-1.5], [sep_loc_1,sep_loc_1], ax=ax, linewidth=sep*sl_no_mplier, color=(0.95,0,0))
    sep_loc_2 = sl_no_mplier*(all_vols.shape[3]+all_vols.shape[2]+sep+sep/2)+sl_no_mplier//2-0.5
    data_linewidth_plot([0.5, psnrs.shape[1]-1.5], [sep_loc_2,sep_loc_2], ax=ax, linewidth=sep*sl_no_mplier, color=(0.95, 0, 0))
    fig.tight_layout()
    plt.show()
    plt.savefig(f'/raid/yesiloglu/data/real_time_volumetric_mri/{pt_id}/temporal_evol_gifs/ax_cr_sg_psnrs_wrt_init_vs_t_sl_{pt_id}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    '''
    # Create and save the plot of PSNRs:
    fig,ax = plt.subplots(1,3)
    psnrs = [calc_psnrs_wrti(all_vols, ax_cr_sg) for ax_cr_sg in [0,1,2]]
    min_psnr = min([ax_cr_sg_psnrs.min() for ax_cr_sg_psnrs in psnrs])
    max_psnr = max([ax_cr_sg_psnrs.max() for ax_cr_sg_psnrs in psnrs])
    for ax_cr_sg in [0,1,2]:
        
        im_type_str = 'axial' if ax_cr_sg == 0 else 'coronal' if ax_cr_sg == 1 else 'sagittal' if ax_cr_sg == 2 else 'ERROR'
        print(f'Creating and saving the plot of {im_type_str.capitalize()} PSNRs.')
        #fig,ax = plt.subplots(figsize=(9,8))
        
        sl_no_mplier = 5
        im = ax[ax_cr_sg].imshow(np.repeat(psnrs[ax_cr_sg], sl_no_mplier, axis=1), vmin=min_psnr, vmax=max_psnr)
        ax[ax_cr_sg].set_title(f'{im_type_str.capitalize()} PSNRs wrt the Initial Image')
        ax[ax_cr_sg].set_xlabel('Slice No')
        ax[ax_cr_sg].set_xticks(sl_no_mplier*np.arange(0,psnrs[ax_cr_sg].shape[1],step=10) + sl_no_mplier//2)
        ax[ax_cr_sg].set_xticklabels(np.arange(0,psnrs[ax_cr_sg].shape[1],step=10))
        ax[ax_cr_sg].set_ylabel('Time Index')
        ax[ax_cr_sg].set_yticks(np.arange(0,psnrs[ax_cr_sg].shape[0],step=50))
        ax[ax_cr_sg].set_yticklabels(np.arange(0,psnrs[ax_cr_sg].shape[0],step=50))
    #fig.colorbar(im, ax=ax, orientation='vertical')
    #fig.colorbar()
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)
    fig.tight_layout()
    plt.show()
    plt.savefig(f'/raid/yesiloglu/data/real_time_volumetric_mri/{pt_id}/temporal_evol_gifs/ax_cr_sg_psnrs_wrt_init_vs_t_sl_{pt_id}.pdf', dpi=150, bbox_inches='tight')
    plt.close(fig)
    '''        
    
if __name__ == "__main__":
    main()