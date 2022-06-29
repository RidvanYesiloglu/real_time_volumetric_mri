import matplotlib.pyplot as plt
import numpy as np

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
    plt.savefig(f'/raid/yesiloglu/data/real_time_volumetric_mri/{pt_id}/temporal_evol_gifs/3dpsnrs_wrt_init', dpi=96, bbox_inches='tight')
    plt.close(fig)
    # Create and save the plot of PSNRs:
    for ax_cr_sg in [0,1,2]:
        psnrs = calc_psnrs_wrti(all_vols, ax_cr_sg)
        im_type_str = 'axial' if ax_cr_sg == 0 else 'coronal' if ax_cr_sg == 1 else 'sagittal' if ax_cr_sg == 2 else 'ERROR'
        print(f'Creating and saving the plot of {im_type_str.capitalize()} PSNRs.')
        fig,ax = plt.subplots(figsize=(9,8))
        sl_no_mplier = 5
        im = ax.imshow(np.repeat(psnrs, sl_no_mplier, axis=1))
        ax.set_title(f'{im_type_str.capitalize()} PSNRs wrt the Initial Image')
        ax.set_xlabel('Slice No')
        ax.set_xticks(sl_no_mplier*np.arange(0,psnrs.shape[1],step=10) + sl_no_mplier//2)
        ax.set_xticklabels(np.arange(0,psnrs.shape[1],step=10))
        ax.set_ylabel('Time Index')
        ax.set_yticks(np.arange(0,psnrs.shape[0],step=50))
        ax.set_yticklabels(np.arange(0,psnrs.shape[0],step=50))
        fig.colorbar(im, ax=ax, orientation='vertical')
        fig.tight_layout()
        plt.show()
        plt.savefig(f'/raid/yesiloglu/data/real_time_volumetric_mri/{pt_id}/temporal_evol_gifs/{im_type_str}_psnrs_wrt_init.pdf', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
if __name__ == "__main__":
    main()