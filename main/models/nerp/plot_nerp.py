import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FormatStrFormatter
def plot_change_of_value(vals, metric_name, repr_str, runno, to_save=False, save_folder=None):
    x1 = np.arange(1,len(vals)+1)
    y1 = np.asarray(vals)
    
    fig, ax1 = plt.subplots(1, 1)
    fig.suptitle(f'Change of {metric_name} Value w.r.t. Epoch No')
    
    ax1.plot(x1, y1)
    ax1.set_ylabel(f'{metric_name}')
    ax1.set_xlabel('Epoch Number')
    plt.show()
    if to_save:
        plt.savefig(os.path.join(save_folder, f'{metric_name}_vs_ep_{runno}.png'),bbox_inches='tight')
    plt.close('all')
    return plt

