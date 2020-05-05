from pyicoshift import Icoshift
from pyicoshift.nmrdatatools.general import plot_spectra
import scipy as sp
import numpy as np
from scipy.io import loadmat
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


if __name__ == '__main__':
    wine = loadmat('../data/WineData.mat')
    X = wine['X']
    data_list = list(X)
    ppm = wine['ppm'][0]
    target = X[35]
    peaks, dict = find_peaks(target, height=int(np.max(target)*0.01))
    intb = []

    last_peak = 0

    offset = 50
    in_cluster_max_dist = 100
    i = 0
    for peak in peaks:
        #if peak >= last_cluster+min_dist:
        #    intb.append(peak-offset)
        #    last_cluster = peak
        #    last_peak = peak
        if last_peak == 0:
            intb.append(peak-offset)
            print('ignore %i' % i)
            i += 1
            last_peak = peak
        if peak > last_peak + in_cluster_max_dist:
            intb.append(last_peak+offset)
            i += 1
            print('ignore %i' % i)
            intb.append(peak-offset)
            i += 1
        last_peak = peak
    intb.append(peak+offset)
    i += 1
    print('ignore %i' % i)
    print(i)
    print(intb)
    print(peaks)
    print(dict)
    fig = plot_spectra([target], interval_borders=intb)
    plt.show()



