import matplotlib.pyplot as plt
import numpy as np

def calc_3dpsnrs_wrti(all_vols):
    return 20*np.log((all_vols[0:1]**2).max((1,2,3)))-10*np.log(((all_vols[0:1]-all_vols)**2).mean((1,2,3)))

def main(all_vols, pt_id):
    # Calculate 3d PSNRs with respect to the initial image
    psnrs = calc_3dpsnrs_wrti(all_vols)
    # Create and save the plot of PSNRs:
    fig,ax = plt.subplots()
    plt.plot(psnrs)
    ax.set_title('3D PSNRs wrt the Initial Image')
    ax.set_xlabel('Time Index')
    ax.set_ylabel('PSNR (dB)')
    plt.show()
    plt.savefig('/raid/yesiloglu/data/real_time_volumetric_mri/{pt_id}/temporal_evol_gifs/3dpsnrs_wrt_init', dpi=96, bbox_inches='tight')
    plt.close()
    
if __name__ == "__main__":
    main()