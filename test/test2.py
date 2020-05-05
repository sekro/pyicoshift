from ..pyicoshift import Icoshift
from ..pyicoshift.functions import plot_spectra
import scipy as sp
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


if __name__ == '__main__':
    wine = loadmat('../data/WineData.mat')

    names = list(range(0,40))
    X = wine['X']
    data_list = list(X)
    ppm = wine['ppm'][0]
    ppms = [ppm for i in range(0,39)]
    ppm_ints = wine['ppm_ints'][0]
    wine_ints = wine['wine_ints'][0]
    custom_intervals = []
    for i in range(0, wine_ints.shape[0], 2):
        custom_intervals.append((int(wine_ints[i]),int(wine_ints[i+1])))
    print(custom_intervals)

    fig1 = plot_spectra(data_list, names, ppms)
    fig2 = plot_spectra(list(X), names, ppms, 1.25, 1.05)
    # plt.figure()
    #for i in range(X.shape[0]):
    #    plt.plot(ppm, X[i,:])
    # plt.show()
    print(X.shape, ppm.shape)

    # plt.figure()
    #for i in range(X.shape[0]):
    #    plt.plot(ppm[7151:7550], X[i,7151:7550])


    lacInter = list(range(7551,7751))
    print(np.issubdtype(X.dtype, np.number))

    # instance icoshift obj

    fix_n_int = Icoshift()


    # assign signals

    fix_n_int.signals = X

    fix_n_int.inter = ('n_intervals', 2)



    # non default configs
    fix_n_int.target = 'maxcorr'


    # run the shifting

    print('---fix n int---')
    fix_n_int.run()
    print(fix_n_int.inter)
    print(fix_n_int._split_list)

    #plot

    fig3 = plot_spectra(list(fix_n_int.result), names, ppms, interval_borders=fix_n_int._split_list)

    plt.show()
