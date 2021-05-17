import os 
import h5py
import numpy as np

from scipy.ndimage.filters import uniform_filter1d
import matplotlib.pyplot as plt


def plot_histogram(all_histograms, save_path, attr_id, attr_name, k=25, smoothing=True):

    x = np.array(list(range(100)))

    if smoothing:
        benign = uniform_filter1d(all_histograms[k][0][attr_id], size=5)
        atypical = uniform_filter1d(all_histograms[k][1][attr_id], size=5)
        malignant = uniform_filter1d(all_histograms[k][2][attr_id], size=5)
    else:
        benign = all_histograms[k][0][attr_id]
        atypical = all_histograms[k][1][attr_id]
        malignant = all_histograms[k][2][attr_id]

    plt.plot(x, benign, label="benign")
    plt.plot(x, atypical, label="atypical")
    plt.plot(x, malignant, label="malignant")
    plt.title(attr_name)
    plt.legend()
    plt.savefig(os.path.join(save_path, attr_name + '.png'))
    plt.clf() 


def h5_to_numpy(h5_path, key):
    h5_object = h5py.File(h5_path, 'r')
    out = np.array(h5_object[key])
    return out
