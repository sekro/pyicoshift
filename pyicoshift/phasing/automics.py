"""
Automics phase correction for 1D NMR data
Apparently rate1 needs to be 270 to obtain results comparable to bruker apk
Adapted from:
https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-10-83
http://code.google.com/p/automics/
by
Sebastian Krossa 08/2019
NTNU Trondheim
sebastian.krossa@ntnu.no

TODO this function is still a bit experimental - the superfast mode works best with data that needs only small
TODO corrections
TODO needs code cleanup
"""

import numpy as np

import math

from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression



def automics_phase_corr_fast(data, p0=0.0, p1=0.0, rate0=0.0, rate1=0.0, break_early=False, superfast=False):
    """

    :param data:
    :param p0:
    :param p1:
    :param rate0:
    :param rate1:
    :param break_early:
    :param superfast:
    :return:
    """
    N = data.shape[0]
    if superfast:
        win_size = 50
        left_1_start = 1200
        right_1_start = N - 1400
    else:
        res_left = find_interval_fast(data, break_early=break_early)
        res_right = find_interval_fast(data, from_right=True, break_early=break_early)
        wins = [0, 0]
        left_1_start, wins[0], _, _ = res_left[0]
        # left_2_start, wins[1], _, _ = res_left[1]
        right_1_start, wins[1], _, _ = res_right[0]
        # right_2_start, wins[3], _, _ = res_right[1]
        win_size = sorted(wins)[0]
    left_2_start = left_1_start + win_size * 4
    right_2_start = right_1_start + win_size * 4
    left_1_stop = left_1_start + win_size
    left_2_stop = left_2_start + win_size
    right_1_stop = right_1_start + win_size
    right_2_stop = right_2_start + win_size

    #print(win_size, (left_1_start, left_1_stop), (left_2_start, left_2_stop), (right_1_start, right_1_stop),
    #      (right_2_start, right_2_stop))
    a = 0.0
    b = 0.0
    c = 0.0
    d = 0.0
    # left = 500
    # right = 500
    for i in range(left_1_start, left_1_stop):
        a += data[i].real
        b += data[i].imag
    for i in range(left_2_start, left_2_stop):
        c += data[i].real
        d += data[i].imag
    p1_error = np.arctan((a - c) / (d - b)) * 180 / np.pi + rate0
    #print(a, b, c, d)
    # if b / (left_1_stop - left_1_start) > d / (left_2_stop - left_2_start):
    # if b > d:
    #    p1_error += 180
    a = 0.0
    b = 0.0
    c = 0.0
    d = 0.0
    for i in range(right_1_start, right_1_stop):
        a += data[i].real
        b += data[i].imag
    for i in range(right_2_start, right_2_stop):
        c += data[i].real
        d += data[i].imag
    p2_error = np.arctan((a - c) / (d - b)) * 180 / np.pi + rate1
    #print(a, b, c, d)
    a = 0.0
    b = 0.0
    c = 0.0
    d = 0.0
    for i in range(left_1_start, left_1_stop):
        a += data[i].real
        b += data[i].imag
    for i in range(right_2_start, right_2_stop):
        c += data[i].real
        d += data[i].imag
    p0_only = np.arctan((a - c) / (d - b)) * 180 / np.pi
    # if b / (right_1_stop - right_1_start) > d / (right_2_stop - right_2_start):
    # if b > d:
    #    p2_error += 180
    i = (left_1_start + left_1_stop) / 2
    j = (left_2_start + left_2_stop) / 2
    k = (2 * N - right_2_start - right_2_stop) / 2
    l = (2 * N - right_1_start - right_1_stop) / 2
    #print(i, j, k, l)
    p1 = 2 * N * (p2_error - p1_error) / (k + l - i - j)
    p0 = p1_error - (i + j) * (p2_error * p1_error) / (k + l - i - j)
    left = left_1_start
    right = N - right_2_start
    # these 2 eq are directly taken from the automics source code - not 100% sure if (N - left - right) is correct
    p1_v = N * (p2_error - p1_error) / (N - left - right)
    p0_v = p1_error - p2_error * (p1_error - p2_error) / (N - left - right)
    #print(p1_error, p2_error, p0, p1, p0_v, p1_v, p0_only)
    return reduce_angle(p0), reduce_angle(p1), reduce_angle(p0_v), reduce_angle(p1_v), reduce_angle(p0_only)


def find_interval_fast(data,
                       significance_lvl=0.05,
                       n_times_snr_cutoff=1.5,
                       st_factor=0.001,
                       mins_factor=0.01,
                       maxs_factor=0.2,
                       scan_iv_factor=0.1,
                       from_right=False,
                       break_early=False):
    """

    :param data:
    :param significance_lvl:
    :param n_times_snr_cutoff:
    :param st_factor:
    :param mins_factor:
    :param maxs_factor:
    :param scan_iv_factor:
    :param from_right:
    :param break_early:
    :return:
    """
    data_size = data.shape[0]
    approx_data_size = round(data_size, -1*int(math.log(data_size, 10)))
    scan_interval = int(round(scan_iv_factor*approx_data_size))
    if scan_interval < 100:
        scan_interval = 100
    step_increment = int(round(st_factor*scan_interval))
    if step_increment <= 0:
        step_increment = 1
    min_step_size = int(round(mins_factor*scan_interval))
    if min_step_size < 100:
        min_step_size = 100
    max_step_size = int(round(maxs_factor*scan_interval)) + step_increment
    if max_step_size < min_step_size:
        max_step_size = 2 * min_step_size
    if from_right:
        upper = data_size
        lower = data_size - scan_interval
    else:
        lower = 0
        upper= lower + scan_interval

    top_interval = []

    for step in range(min_step_size, max_step_size, step_increment):
        if from_right:
            lower=upper-math.ceil(scan_interval/step)*step-step
        slope_rank = []
        for i in range(lower,upper,int(step/4)):
            if i+step > data_size:
                break
            x = np.array(range(0,step)).reshape((-1, 1))
            model = LinearRegression().fit(x, data[i:i+step].real)
            slope_rank.append((abs(float(model.coef_)), i))
            if break_early and model.coef_ <= 0.5 * abs(np.mean(data[lower:upper].real)/np.std(data[lower:upper].real)):
                break
        top_slope, top_start_index = sorted(slope_rank)[0]
        res = adfuller(data[top_start_index:top_start_index+step].real)
        if top_slope <= n_times_snr_cutoff * abs(np.mean(data[lower:upper].real)/np.std(data[lower:upper].real)) and res[1] <= significance_lvl:
            top_interval.append((top_start_index, step, top_slope, res[1]))
    print(n_times_snr_cutoff * np.mean(data[lower:upper].real)/np.std(data[lower:upper].real))
    print(sorted(top_interval, key=lambda x: x[2])[0:3])
    print(sorted(top_interval, key=lambda x: x[1], reverse=True)[0:3])
    return sorted(top_interval, key=lambda x: x[2])


def reduce_angle(p):
    if p <= -360:
        p = -1 * (abs(p) % 360)
    elif p >= 360:
        p = p % 360
    return p

