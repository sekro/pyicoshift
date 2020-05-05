"""
 - general nmr data tools/functions -

Sebastian Krossa 08/2019
NTNU Trondheim
sebastian.krossa@ntnu.no
"""

import matplotlib.pyplot as plt
import numpy as np
import logging


def shift_bit_length(x):
    """
    returns closest to the power of 2 number bigger than x
    """
    return 1<<(x-1).bit_length()


def plot_spectra(data_list, names=None, ppms_list=None, start=None, stop=None, interval_borders=None, title=None):
    """
    Plots 1D NMR spectra
    adapted from https://www.mfitzp.com/article/1d-1h-nmr-data-processing/

    :param names:
    :param ppms_list: data_pts to ppm map
    :param data_list: list of data_sets to plot in one figure
    :param start: start/left ppm (optional)
    :param stop:  stopt/right ppm (optional)
    :param interval_borders: draws vertical lines at provided points
    :param title: figure title
    :return: the figure obj
    """
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 1, 1)
    if names is None:
        names = list(range(0, len(data_list)))
    if ppms_list is not None:
        # in case input for ppms_list was just one map:
        if not isinstance(ppms_list, list):
            # generate a list of ppms maps of same size as data_list
            ppms_list = [ppms_list for i in data_list]
        for data, ppms, name in zip(data_list, ppms_list, names):
            if start:
                ixs = min(range(len(ppms)), key=lambda i: abs(ppms[i]-start))
                ppms = ppms[ixs:]
                data = data[ixs:]
            if stop:
                ixs = min(range(len(ppms)), key=lambda i: abs(ppms[i]-stop))
                ppms = ppms[:ixs]
                data = data[:ixs]
            ax.plot(ppms, data.real, label=name)
        ax.set_xlabel('ppm')
        ax.invert_xaxis()
    else:
        for data, name in zip(data_list, names):
            ax.plot(range(0, data.shape[0]), data.real, label=name)
        ax.set_xlabel('data points')
    if interval_borders is not None:
        if ppms_list is not None:
            interval_borders = [ppms_list[0][x] for x in interval_borders]
        logging.debug('using interval borders %s' % str(interval_borders))
        y_limit = ax.get_ylim()
        x_lines = np.repeat(np.array(interval_borders), repeats=3)
        y_lines = np.repeat(np.array([[y_limit[0], y_limit[1], np.nan]]), repeats=len(interval_borders),
                            axis=0).flatten()
        logging.debug('x vals %s' % str(x_lines))
        logging.debug('y vals %s' % str(y_lines))
        ax.plot(x_lines, y_lines, scalex=False)
    ax.legend()
    if isinstance(title, str):
        ax.set_title(title)
    return fig


def get_outlier_mask(input_vector, outlier_iqr_factor):
    upper_quartile = np.percentile(input_vector, 75)
    lower_quartile = np.percentile(input_vector, 25)
    outlier_cuttoff = (upper_quartile - lower_quartile) * outlier_iqr_factor
    cutoffs = (lower_quartile - outlier_cuttoff, upper_quartile + outlier_cuttoff)
    return np.logical_or(input_vector <= cutoffs[0], input_vector >= cutoffs[1])

