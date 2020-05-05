"""
- bruker nmr data helper / convenience functions -

Sebastian Krossa 08/2019
NTNU Trondheim
sebastian.krossa@ntnu.no
"""

import numpy as np
import nmrglue as ng
from .general import shift_bit_length

def get_bruker_fid_ppms(dic, size):
    # adapted from https://www.mfitzp.com/article/1d-1h-nmr-data-processing/
    offset = (float(dic['acqus']['SW']) / 2) - (float(dic['acqus']['O1']) / float(dic['acqus']['BF1']))
    start = float(dic['acqus']['SW']) - offset
    end = -offset
    step = float(dic['acqus']['SW']) / size

    return np.arange(start, end, -step)[:size]


def get_data_pt_from_ppm(ppms, ppm):
    return min(enumerate(list(ppms)), key=lambda x: abs(x[1]-ppm))


def get_bruker_pk_angles(dic):
    """
    returns PHC0 and PHC1 as tuple - with some adjustments
    -> seems that a PHC1 inversion with a data centered pivot point is needed
    -> have to do PHC0-180 and PHC1-360 in general dunno why
    """
    return (float(dic['procs']['PHC0']), float(dic['procs']['PHC1']))


def get_bruker_lb_for_em_in_points(dic):
    return float(dic['procs']['LB'])/float(dic['acqus']['SW_h'])


def load_bruker_fid(bruker_folder, remove_digital_filter=True, trunc_fid=0, line_broadening=None):
    """

    :param bruker_folder: string containing path to bruker fid folder
    :return: the data as numpy array and the ppms to data point mapper as numpy array
    """

    folder = bruker_folder
    dic, fid = ng.bruker.read(folder)
    if line_broadening is not None:
        lb = line_broadening / float(dic['acqus']['SW_h'])
    else:
        lb = get_bruker_lb_for_em_in_points(dic)
    zero_fill_size = shift_bit_length(fid.shape[0])
    if remove_digital_filter:
        fid = ng.bruker.remove_digital_filter(dic, fid, truncate=False)
    fid = ng.process.proc_base.em(fid, lb=lb)
    fid = ng.proc_base.zf_size(fid, zero_fill_size)  # <2>
    fid = ng.proc_base.rev(fid)  # <3>
    if trunc_fid:
        data = np.array(ng.proc_base.fft(fid[trunc_fid:]))
        print('truncated fid')
    else:
        data = np.array(ng.proc_base.fft(fid))
    # a bit strange that i ll have to do that --> check if only for this test dataset or all from bruker
    data = ng.proc_base.ps(data, p0=(-180), p1=(-360), inv=False)
    ppms = get_bruker_fid_ppms(dic, data.shape[0])
    return (data, ppms)

# pseudo auto baseline functions for HRMAS NMR data

def auto_baseline(data, ppms):
    node_list = [get_data_pt_from_ppm(ppms, 17.5)[0], get_data_pt_from_ppm(ppms, 15)[0],
                 get_data_pt_from_ppm(ppms, 10)[0], get_data_pt_from_ppm(ppms, 8.75)[0],
                 get_data_pt_from_ppm(ppms, 8.46)[0], get_data_pt_from_ppm(ppms, 8.3)[0],
                 get_data_pt_from_ppm(ppms, 8)[0], get_data_pt_from_ppm(ppms, 7.25)[0],
                 get_data_pt_from_ppm(ppms, -2.5)[0], get_data_pt_from_ppm(ppms, -5)[0],
                 get_data_pt_from_ppm(ppms, -7.5)[0]
                 ]
    return ng.process.proc_bl.base(data, nl=node_list)

def auto_baseline2(data, ppms):
    node_list = [get_data_pt_from_ppm(ppms, 17.5)[0], get_data_pt_from_ppm(ppms, 15)[0],
                 get_data_pt_from_ppm(ppms, 10)[0], get_data_pt_from_ppm(ppms, 8.75)[0],
                 get_data_pt_from_ppm(ppms, 8.455)[0], get_data_pt_from_ppm(ppms, 8.42)[0],
                 get_data_pt_from_ppm(ppms, 8)[0], get_data_pt_from_ppm(ppms, 7.25)[0],
                 get_data_pt_from_ppm(ppms, 6.15)[0], get_data_pt_from_ppm(ppms, 5.7)[0],
                 get_data_pt_from_ppm(ppms, 4.67)[0], get_data_pt_from_ppm(ppms, 4.17)[0],
                 get_data_pt_from_ppm(ppms, 2.88)[0], get_data_pt_from_ppm(ppms, 1.84)[0],
                 get_data_pt_from_ppm(ppms, 0)[0], get_data_pt_from_ppm(ppms, -1.5)[0],
                 get_data_pt_from_ppm(ppms, -2.5)[0], get_data_pt_from_ppm(ppms, -5)[0],
                 get_data_pt_from_ppm(ppms, -7.5)[0]
                 ]
    return ng.process.proc_bl.base(data, nl=node_list)

def auto_baseline3(data, ppms):
    node_list = [get_data_pt_from_ppm(ppms, 12)[0], get_data_pt_from_ppm(ppms, 9)[0],
                 get_data_pt_from_ppm(ppms, 8.45)[0], get_data_pt_from_ppm(ppms, 8.35)[0],
                 get_data_pt_from_ppm(ppms, 6.25)[0], get_data_pt_from_ppm(ppms, 4.68)[0],
                 get_data_pt_from_ppm(ppms, 4.174)[0], get_data_pt_from_ppm(ppms, 4.0)[0],
                 get_data_pt_from_ppm(ppms, 2.88)[0], get_data_pt_from_ppm(ppms, 2.75)[0],
                 get_data_pt_from_ppm(ppms, 2.6)[0], get_data_pt_from_ppm(ppms, 2.46)[0],
                 get_data_pt_from_ppm(ppms, 2.18)[0], get_data_pt_from_ppm(ppms, 1.84)[0],
                 get_data_pt_from_ppm(ppms, 1.51)[0], get_data_pt_from_ppm(ppms, 1.06)[0],
                 get_data_pt_from_ppm(ppms, 0)[0], get_data_pt_from_ppm(ppms, -2)[0],
                 get_data_pt_from_ppm(ppms, -7.5)[0]
                 ]
    return ng.process.proc_bl.base(data, nl=node_list)
