"""
pyicoshift
from scratch python implementation of the
"icoshift - interval Correlation Optimized shifting" algorithm for peak alignment
Sebastian Krossa 08/2019
NTNU Trondheim
sebastian.krossa@ntnu.no

icoshift by Authors:
    Francesco Savorani - Department of Food Science
                         Quality & Technology - Spectroscopy and Chemometrics group
                         Faculty of Sciences
                         University of Copenhagen - Denmark
    email: frsa@life.ku.dk - www.models.life.ku.dk
    Giorgio Tomasi -     Department of Basic Science and Environment
                         Soil and Environmental Chemistry group
                         Faculty of Life Sciences
                         University of Copenhagen - Denmark
    email: giorgio.tomasi@ec.europa.eu - www.igm.life.ku.dk

"""
import time
from typing import Union, List, Tuple
from .functions import get_index_signal_with_highest_correlation, get_avg2_target, get_max_signal_idx, \
    get_fill_matrix, shift_signals, ccfftshift
from .nmrdatatools.general import plot_spectra
from scipy.signal import find_peaks
import concurrent.futures
import numpy as np
import logging


# logging.basicConfig(level=logging.DEBUG)


class Icoshift:
    def __init__(self):
        self._loglvl = logging.INFO
        self._loglvl_available = {'critical': 50,
                                  'error': 40,
                                  'warn': 30,
                                  'info': 20,
                                  'debug': 1}
        logging.basicConfig(level=self._loglvl)
        logging.basicConfig(style='%', format="%(name)s -> %(levelname)s: %(message)s")
        self._name = None
        self._target: np.ndarray = None
        self._target_mode: str = 'average'
        self._used_signal_n_as_target: int = None
        self._target_in_intervals: List[np.ndarray] = None
        self._signals: np.ndarray = None
        self._signals_in_intervals: List[np.ndarray] = None
        self._signal_length: int = None
        self._n_signals: int = None
        self._shifted_signals: np.ndarray = None
        self._shifted_signals_in_intervals: List[np.ndarray] = None
        self._signal_names: List[str] = None
        self._n_intervals: int = None
        self._fixed_interval_length: int = None
        # list of tuple(start, end)
        self._intervals: List[Tuple[Union[int, float], Union[int, float]]] = None
        self._split_list: List[int] = None
        self._ignore_intervals_no: List[int] = []
        # list of points corresponding to the respective value in units for each datapoint
        self._unit_to_sample_point_map: np.ndarray = None
        self._unit_name: str = None
        self._align_mode: str = 'whole'
        self._reconstruction_mode: str = 'adjacent'
        self._inputs_in_datapoints: bool = True
        self._max_shift_correction: int = None
        self._max_shift_correction_mode: str = 'fast'
        self._pre_align_globally: bool = False
        self._max_shift_find_iterations: int = 1000
        self._best_shift_find_increment: int = 1
        self._fast_shift_find_increment: int = 5
        self._avg2factor: int = 3
        self._fig_display_interval_borders = True
        # Warning: this is an experimental feature!
        self._parallel_interval_processing = False
        self._use_threads = True
        self._align_auto_pick_min_peak_height_factor: float = 0.005
        self._align_auto_pick_max_cluster_dist_factor = 0.02
        self._align_auto_pick_offset_factor = 0.01
        self._align_auto_pick_ignore_intervals_between_peak_cluster = True
        self._fig_pre_align = None
        self._fig_post_align = None
        self._fig_target = None
        self._target_mode_switch = {'average': self._target_average,
                                    'median': self._target_median,
                                    'max': self._target_max,
                                    'average2': self._target_average2,
                                    'maxcorr': self._target_max_correlation,
                                    'user_defined': self._target_user_defined
                                    }
        self._align_modes_switch = {'whole': self._align_whole,
                                    'n_intervals': self._align_n_intervals,
                                    'fixed_interval_length': self._align_fixed_interval_length,
                                    'user_defined_intervals': self._align_user_defined_intervals,
                                    'shift_relative_to_region': self._align_relative_to_region,
                                    'auto_pick': self._align_auto_pick_intervals
                                    }
        self._reconstruction_modes_available = ['nan', 'zero', 'adjacent']
        self._max_shift_correction_mode_switch = {'manual': self._max_shift_manual,
                                                  'best': self._max_shift_best,
                                                  'fast': self._max_shift_fast
                                                  }
        self._input_type_switch = {'p': True,
                                   'pts': True,
                                   'points': True,
                                   '1': True,
                                   'u': False,
                                   'units': False,
                                   'ppm': False,
                                   '0': False}
        self._flags = {'max': False,
                       'avg2': False}

    @property
    def name(self) -> str:
        if self._name:
            return self._name
        else:
            return 'no name'

    @name.setter
    def name(self, value: str):
        if isinstance(value, str):
            self._name = value
        else:
            raise ValueError('name has to be of type string')

    @property
    def global_pre_align(self) -> bool:
        return self._pre_align_globally

    @global_pre_align.setter
    def global_pre_align(self, value: bool):
        if isinstance(value, bool):
            self._pre_align_globally = value
        else:
            raise ValueError('only bool allowed')

    @property
    def unit_vector(self) -> np.ndarray:
        return self._unit_to_sample_point_map

    @unit_vector.setter
    def unit_vector(self, value: np.ndarray):
        if self._signal_length:
            if isinstance(value, np.ndarray):
                if value.ndim == 1:
                    if value.shape[0] == self._signal_length:
                        self._unit_to_sample_point_map = value
                    else:
                        raise ValueError('Length of unit vector does not match sample length')
                else:
                    raise ValueError('Dimension of unit vector is not 1')
            else:
                raise ValueError('Unit vector has to be of type numpy.ndarray')
        else:
            raise ValueError('Provide sample vectors first')

    @property
    def signal_names(self) -> List[str]:
        return self._signal_names

    @signal_names.setter
    def signal_names(self, value: List[str]):
        if isinstance(value, list):
            if all(isinstance(x, str) for x in value):
                self._signal_names = value
            else:
                raise ValueError('All sample signal_names have to be of type str')
        else:
            raise ValueError('Sample signal_names have to be provided as list of str')

    @property
    def input_type(self) -> str:
        if self._inputs_in_datapoints:
            return 'Input all intervals in datapoints'
        else:
            return 'Input all intervals in units'

    @input_type.setter
    def input_type(self, value: Union[bool, str, int]):
        if isinstance(value, bool):
            self._inputs_in_datapoints = value
        elif isinstance(value, int):
            if abs(value) >= 1:
                self._inputs_in_datapoints = True
            else:
                self._inputs_in_datapoints = False
        elif isinstance(value, str):
            if value in self._input_type_switch:
                self._inputs_in_datapoints = self._input_type_switch[value]
            else:
                raise ValueError('Error - invalid input value')
        else:
            raise ValueError('Error - invalid input value')

    @property
    def result(self) -> np.ndarray:
        return self._shifted_signals

    @property
    def signals(self) -> np.ndarray:
        return self._signals

    @signals.setter
    def signals(self, value):
        if self._signals is not None:
            logging.warning('overwriting existing signals and deleting target')
            self._signals = None
            self._signal_length = None
            self._target = None
            self._target_mode = 'average'
        if isinstance(value, list):
            if all(isinstance(x, np.ndarray) for x in value):
                if all(x.ndim == 1 and np.issubdtype(x.dtype, np.number) for x in value):
                    if all(value[0].shape[0] == x.shape[0] for x in value):
                        if self._signal_length:
                            logging.warning('sample length already set - you have most likely assigned a manual '
                                            'target vector allready')
                            if self._signal_length == value[0].shape[0]:
                                logging.info('Target and sample vector length match - all good')
                                self._signals = np.array(value)
                                self._n_signals = len(value)
                            else:
                                raise ValueError('Target and sample vector length do not match')
                        else:
                            self._signals = np.array(value)
                            self._signal_length = value[0].shape[0]
                    else:
                        raise ValueError('Input sample vectors have to be of same length')
                else:
                    raise ValueError('Each input sample vector has to be of dimension 1 and dtype must be child of '
                                     'number')
            else:
                raise ValueError('Input sample vectors have to be provided as numpy.ndarrays')
        elif isinstance(value, np.ndarray):
            if value.ndim == 2 and np.issubdtype(value.dtype, np.number):
                if self._signal_length:
                    logging.warning('sample length already set - you have most likely assigned a manual target vector '
                                    'allready')
                    if self._signal_length == value.shape[1]:
                        logging.info('Target and sample vector length match - all good')
                        self._signals = value
                        self._n_signals = value.shape[0]
                    else:
                        raise ValueError('Target and sample vector length do not match')
                else:
                    self._signals = value
                    self._n_signals, self._signal_length = value.shape
            else:
                raise ValueError('Dimension of the numpy.ndarray inccorect or non-numeric')
        else:
            raise ValueError('Input has to be a list of sample vectors as numpy.ndarray or 2 dimensional numpy.ndarray')

    @property
    def inter(self) -> List[Tuple[Union[int, float], Union[int, float]]]:
        return self._intervals

    @inter.setter
    def inter(self, value: Union[str, List[Tuple[int, int]], List[Tuple[float, float]],
                                 Tuple[str, Union[int, float, Tuple[int, int], Tuple[float, float]]]]):
        if self._inputs_in_datapoints:
            allowed_types = int
        elif self._unit_to_sample_point_map is not None:
            allowed_types = (int, float)
        else:
            raise RuntimeError('Inputs in units set but no unit to data points map set')
        if isinstance(value, list):
            # TODO: check for correct order of each start and stop
            if all(isinstance(x, tuple) and len(x) == 2 for x in value):
                if all(isinstance(x, allowed_types) and isinstance(y, allowed_types) for x, y in value):
                    if self._inputs_in_datapoints:
                        self._intervals = value
                    else:
                        self._intervals = [(self._convert_to_datapts(x), self._convert_to_datapts(y)) for x, y in value]
                    self._align_mode = 'user_defined_intervals'
                else:
                    raise ValueError('List of of intervals have to be of type: ', allowed_types)
            else:
                raise ValueError('List have to hold tuples of size 2')
        elif isinstance(value, tuple):
            if len(value) == 2:
                option, val = value
                if isinstance(val, tuple):
                    if all(isinstance(x, allowed_types) for x in val) and option == 'shift_relative_to_region':
                        if self._inputs_in_datapoints:
                            self._intervals = [val]
                        else:
                            self._intervals = [(self._convert_to_datapts(val[0]), self._convert_to_datapts(val[1]))]
                        self._align_mode = option
                    else:
                        raise ValueError('Error - invalid value for inter')
                elif isinstance(val, allowed_types):
                    if option == 'n_intervals' and isinstance(val, int):
                        self._n_intervals = val
                        self._align_mode = option
                    elif option == 'fixed_interval_length':
                        if self._inputs_in_datapoints:
                            self._fixed_interval_length = val
                        else:
                            self._fixed_interval_length = self._convert_length_to_datapts(val)
                        self._align_mode = option
                    else:
                        raise ValueError('Error - invalid value for inter')
                else:
                    raise ValueError('Error - invalid value for inter')
            else:
                raise ValueError('Only accept tuples of size 2')
        elif isinstance(value, str):
            if value in ['whole', 'auto_pick']:
                self._align_mode = value
            else:
                raise ValueError('Error - invalid value for inter')
        else:
            raise ValueError('Input "whole" or provide custom intervals as list or tuple (option[str], ref interval '
                             'as list[int/float] or no/length of intervals as int/float)')

    @property
    def target(self) -> Tuple[str, Union[np.ndarray, list]]:
        if self._used_signal_n_as_target is not None:
            if self._signal_names is not None:
                target_description = 'mode <%s> using sample %s as target' % (
                    self._target_mode, self._signal_names[self._used_signal_n_as_target])
            else:
                target_description = 'mode <%s> using sample no %i as target' % (
                    self._target_mode, self._used_signal_n_as_target)
        else:
            target_description = 'mode <%s>' % self._target_mode
        if self._target_in_intervals:
            return target_description, np.concatenate(self._target_in_intervals)
        else:
            return target_description, self._target

    @target.setter
    def target(self, value: Union[np.ndarray, str]):
        if self._target is not None:
            print('Warning: overwriting existing target')
        if isinstance(value, np.ndarray):
            if value.ndim == 1:
                if self._signal_length == value.shape[0]:
                    self._target = value
                    self._target_mode = 'user_defined'
                elif self._signal_length:
                    self._target = value
                    self._target_mode = 'user_defined'
                    self._signal_length = value.shape[0]
                else:
                    raise ValueError('Length of target vector does not match sample vectors')
            else:
                raise ValueError('Target vector has more than 1 dimension')
        elif isinstance(value, str):
            if value in self._target_mode_switch:
                self._target_mode = value
                self._target = None
            else:
                raise ValueError('Error - invalid value for target')
        else:
            raise ValueError('Target has to be a 1-dimensional numpy array or option as string')

    @property
    def avg2factor(self) -> int:
        return self._avg2factor

    @avg2factor.setter
    def avg2factor(self, value: int):
        if isinstance(value, int):
            if value > 0:
                self._avg2factor = value
            else:
                raise ValueError('Factor for average2 target mode has to be > 0')
        else:
            raise ValueError('Factor for average2 target mode has to be an integer > 0')

    @property
    def max_shift(self) -> Tuple[str, int]:
        return self._max_shift_correction_mode, self._max_shift_correction

    @max_shift.setter
    def max_shift(self, value: Union[str, int, float]):
        if isinstance(value, str):
            if value == 'manual':
                raise ValueError('For manual max shift input the value instead')
            elif value in self._max_shift_correction_mode_switch:
                self._max_shift_correction_mode = value
            else:
                raise ValueError('Error - invalid value for max_shift')
        elif isinstance(value, int) and self._inputs_in_datapoints or \
                isinstance(value, (int, float)) and not self._inputs_in_datapoints:
            self._max_shift_correction_mode = 'manual'
            self._max_shift_correction = value
        else:
            raise ValueError('Error - invalid value for max_shift, has to be an option as string or abs value')

    @property
    def fill_mode(self) -> str:
        return self._reconstruction_mode

    @fill_mode.setter
    def fill_mode(self, value):
        if isinstance(value, str) and value in self._reconstruction_modes_available:
            self._reconstruction_mode = value
        else:
            raise ValueError('Error - invalid value for fill_mode, has to one of ',
                             self._reconstruction_modes_available)

    @property
    def loglvl(self) -> str:
        return logging.getLevelName(self._loglvl)

    @loglvl.setter
    def loglvl(self, value: str):
        if isinstance(value, str):
            if value in self._loglvl_available:
                if value != self._loglvl:
                    self._loglvl = value
                    logging.basicConfig(level=self._loglvl)
            else:
                raise ValueError('Unknown log level')
        else:
            raise ValueError('input has to be of type string')

    def run(self):
        """
        Main function to run the actual icoshift
        1 - pre run coshift whole dataset?
          - co shift / no co shift
        2 - split into intervals
        3 - co shift each interval
        4 - reconstruct from intervals
        """

        next_step = True

        if self._pre_align_globally:
            logging.info('pre-aligning whole signals')
            logging.info('Starting get targets for pre-align')
            next_step = self._target_mode_switch.get(self._target_mode)()
            logging.info('Finished get targets for pre-align')
            if next_step:
                next_step = self._coshift(pre_align=True)
                self._signals = self._shifted_signals
            else:
                raise RuntimeError('Something went wrong during pre-alignment')
            # remove the target used during pre-alignment
            self._target = None
            logging.info('pre-aligning whole signal done')

        if next_step:
            logging.info('Starting splitting into intervals')
            next_step = self._split_signals()
            logging.info('Finished splitting into intervals')
        else:
            raise RuntimeError('Something went wrong during pre-processing')
        if next_step:
            logging.info('Starting get targets')
            next_step = self._target_mode_switch.get(self._target_mode)()
            logging.info('Finished get targets')
        else:
            raise RuntimeError('Something went wrong during splitting into Intervals')
        if next_step:
            self._coshift()

    def make_figures(self):
        if self._fig_display_interval_borders:
            borders = self._split_list
        else:
            borders = None
        self._fig_pre_align = plot_spectra(list(self._signals),
                                           self._signal_names,
                                           self._unit_to_sample_point_map,
                                           interval_borders=borders,
                                           title='%s: All spectra prior shifting' % self.name)
        self._fig_post_align = plot_spectra(list(self._shifted_signals),
                                            self._signal_names,
                                            self._unit_to_sample_point_map,
                                            interval_borders=borders,
                                            title='%s: All spectra after shifting' % self.name)
        self._fig_target = plot_spectra([self.target[1]],
                                        ['Target: ' + self.target[0]],
                                        self._unit_to_sample_point_map,
                                        interval_borders=borders,
                                        title='%s: Target used for shifting' % self.name)

    def _find_and_apply_shift(self, target, signals):
        shifts, scores, n = self._max_shift_correction_mode_switch.get(self._max_shift_correction_mode)(target, signals)
        return shift_signals(signals, shifts, fill=get_fill_matrix(signals, self._reconstruction_mode))

    def _coshift(self, pre_align=False):
        if self._align_mode != 'whole' and not pre_align and self._signals_in_intervals is not None:
            if self._target_in_intervals is not None:
                if len(self._target_in_intervals) == len(self._signals_in_intervals):
                    logging.info('Target and signals intervals in place')
                    shifts = None
                    self._shifted_signals_in_intervals = []
                    # loop through all signals/target intervals
                    # Test if it is really faster using parallel code - if self._use_threads is True code is executed
                    # in parallel threads otherwise in processes
                    # makes maybe only sense for "best" mode - threads seem to be a bit faster than processes, but if
                    # using more threads/processes than cores available no speed up with numpy v1.17
                    # CPU load definitely higher with processes, but the overhead from switching seems not worth it for
                    # smaller data sets
                    # So this feature is is "experimental"
                    # TODO: add chunking into blocks of processes matching available cores
                    # parallel processing does not make sense in case of just one reference interval
                    if self._parallel_interval_processing and self._align_mode != 'shift_relative_to_region':
                        tasks = []
                        if self._use_threads:
                            poolexecutor = concurrent.futures.ThreadPoolExecutor()
                        else:
                            poolexecutor = concurrent.futures.ProcessPoolExecutor()
                        with poolexecutor as executor:
                            logging.info('starting parallel calculation of shifts')
                            start = time.time()
                            for i, (target, signals) in enumerate(
                                    zip(self._target_in_intervals, self._signals_in_intervals)):
                                if i not in self._ignore_intervals_no:
                                    if self._use_threads:
                                        logging.info('starting a thread')
                                    else:
                                        logging.info('starting a process')
                                    future = executor.submit(self._find_and_apply_shift, target, signals)
                                    tasks.append((i, future))
                                else:
                                    tasks.append((i, signals))
                            logging.info('waiting for all threads/processes to finish')
                        logging.info('all threads/processes finished - it took %s s - rebuilding signals now' % str(
                            time.time() - start))
                        start = time.time()
                        for i, item in sorted(tasks):
                            # item is a future obj in case i not in ignore list else its a signal interval
                            if i not in self._ignore_intervals_no:
                                self._shifted_signals_in_intervals.append(item.result())
                            else:
                                self._shifted_signals_in_intervals.append(item)
                        logging.info('signals rebuild - it took %s s' % str(time.time() - start))
                    else:
                        logging.info('starting sequential code execution')
                        start = time.time()
                        for i, (target, signals) in enumerate(
                                zip(self._target_in_intervals, self._signals_in_intervals)):
                            # only calc shift if interval index is not on ignore list
                            if i not in self._ignore_intervals_no:
                                shifts, scores, n = self._max_shift_correction_mode_switch.get(
                                    self._max_shift_correction_mode)(
                                    target, signals)
                                logging.debug('found shifts with [shifts, scores, n]: ', str(shifts), str(scores),
                                              str(n))
                                # only shift interval and add now if not reference region align
                                if self._align_mode != 'shift_relative_to_region':
                                    self._shifted_signals_in_intervals.append(
                                        shift_signals(signals, shifts,
                                                      fill=get_fill_matrix(signals, self._reconstruction_mode))
                                    )
                            else:
                                # add unshifted interval only if not reference region align
                                if self._align_mode != 'shift_relative_to_region':
                                    self._shifted_signals_in_intervals.append(signals)
                        logging.info(
                            'sequential and concatenating results finished - it took %s s' % str(time.time() - start))

                    if self._align_mode != 'shift_relative_to_region':
                        # concatenate to final np.ndarray
                        self._shifted_signals = np.concatenate(self._shifted_signals_in_intervals, axis=1)
                    else:
                        # in case of reference shift use non interval signals and shift with shifts from above
                        self._shifted_signals = shift_signals(self._signals, shifts, fill=get_fill_matrix(
                            self._signals, self._reconstruction_mode))
                    logging.info('All intervals aligned')
                else:
                    raise RuntimeError('Taget and signals intervals amount do not match')
            else:
                raise RuntimeError('Found signal intervals but no target intervals')
        elif self._align_mode == 'whole' or pre_align:
            logging.info('whole signal align mode starting')
            if self._shifted_signals is not None:
                logging.warning('shifted signals present - overwriting previous')
            shifts, scores, n = self._max_shift_correction_mode_switch.get(self._max_shift_correction_mode)(
                self._target,
                self._signals)
            logging.debug('found shifts with [shifts, scores, n]: %s %s %s' % (str(shifts), str(scores), str(n)))
            self._shifted_signals = shift_signals(self._signals, shifts,
                                                  fill=get_fill_matrix(self._signals, self._reconstruction_mode))
            logging.info('signals aligned')
        else:
            raise RuntimeError('called _coshift with undefined behavior')
        return True

    def _run_global_align(self):
        if self._shifted_signals is not None:
            logging.warning('shifted signals present - overwriting previous')
        shifts, scores = ccfftshift(self._target, self._signals)
        self._shifted_signals = shift_signals(self._signals, shifts,
                                              fill=get_fill_matrix(self._signals, self._reconstruction_mode))
        return True

    # Interval functions

    def _split_signals(self):
        if self._align_mode != 'whole':
            split_list_generator = self._align_modes_switch.get(self._align_mode)
            self._split_list = split_list_generator()
            self._signals_in_intervals = np.split(self._signals, self._split_list, axis=1)
        return True

    # target functions

    def _split_target(self):
        if self._align_mode != 'whole':
            split_list_generator = self._align_modes_switch.get(self._align_mode)
            self._target_in_intervals = np.split(self._target, split_list_generator(), axis=0)
            logging.info('target splitted into intervals')
        return True

    def _target_average(self):
        self._target = np.nanmean(self._signals, axis=0)
        logging.info('using mean signal as target')
        self._split_target()
        return True

    def _target_median(self):
        self._target = np.nanmedian(self._signals, axis=0)
        logging.info('using median signal as target')
        self._split_target()
        return True

    def _target_max(self):
        # get the signal in the signals matrix with the highest signal and use this as reference
        if self._signals_in_intervals is not None:
            self._target_in_intervals = []
            for signal_interval in self._signals_in_intervals:
                self._target_in_intervals.append(signal_interval[get_max_signal_idx(signal_interval), :])
            logging.info('using per interval max signal as target')
        else:
            self._target = self._signals[get_max_signal_idx(self._signals), :]
            logging.info('using signal with highest overall signal intensity as target')
        return True

    def _target_average2(self):
        # in the original matlab implementation this seems to be calculated as:
        # target = (mean(target) - min(mean(target)) * factor[default=3] for each interval separately
        # why separately? -> would only affect the "baseline" correction part min(mean(target)) this should not
        # be to important if all input signals are baseline corrected and factor not unreasonably high
        # nevertheless implemented as in matlab version
        if self._signals_in_intervals is not None:
            self._target_in_intervals = []
            for signal_interval in self._signals_in_intervals:
                self._target_in_intervals.append(get_avg2_target(signal_interval, self._avg2factor))
            logging.info('using per interval average2 with a factor of %i' % self._avg2factor)
        else:
            self._target = get_avg2_target(self._signals, self._avg2factor)
            logging.info('using average2 of whole signal as target')
        return True

    def _target_max_correlation(self):
        # This is new and not in the original icoshift implementation
        # calculate the correlation matrix for all signals and select the one that has the highest correlation with all
        idx, signal = get_index_signal_with_highest_correlation(self._signals)
        self._used_signal_n_as_target = idx
        logging.info('signal no. %i has the highest correlation with rest of signals - using it as target' % idx)
        # use that signal as target
        self._target = signal
        # split target into intervals if needed
        self._split_target()
        return True

    def _target_user_defined(self):
        if self._target is not None:
            logging.info('using user-defined target')
            self._split_target()
            return True
        else:
            return False

    # align mode functions

    def _align_whole(self):
        # jaja - whats the point ;)
        return []

    def _align_n_intervals(self):
        size = self._signal_length // self._n_intervals
        n_size_plus_one = self._signal_length % self._n_intervals
        if n_size_plus_one > 0:
            split_list = list(range(size + 1, (n_size_plus_one + 1) * (size + 1), size + 1)) + \
                         list(range(n_size_plus_one * (size + 1) + size, self._signal_length, size))
        else:
            split_list = list(range(size, self._signal_length, size))
        return split_list

    def _align_fixed_interval_length(self):
        if self._signal_length % self._fixed_interval_length != 0:
            # if last interval < fixed_interval_length -> merge last interval with rest
            return list(range(self._fixed_interval_length, self._signal_length, self._fixed_interval_length))[:-1]
        else:
            return list(range(self._fixed_interval_length, self._signal_length, self._fixed_interval_length))

    def _align_user_defined_intervals(self):
        self._ignore_intervals_no = []
        split_list = []
        last_pos = 0
        i = 0
        for start, stop in sorted(self._intervals):
            if start > last_pos:
                self._ignore_intervals_no.append(i)
                split_list.append(start)
                i += 1
                last_pos = start
            if stop < self._signal_length:
                split_list.append(stop)
                i += 1
                last_pos = stop
        if last_pos < self._signal_length - 1:
            self._ignore_intervals_no.append(i)
        return split_list

    def _align_relative_to_region(self):
        return self._align_user_defined_intervals()

    def _align_auto_pick_intervals(self):
        # This function is not implemented in original matlab version of icoshift
        # picks intervals for alignment based on:
        # peak-clusters found in the signal with the highest sum of correlation coef with other signals
        # peak find uses a signal height cutoff off max signal * height_factor
        # peaks belong to clusters if peak distance < signal_length * cluster_dist_factor
        # each cluster gets an offset = signal_length * offset_fact added left and right -> intervals with peaks
        # if ignore_intervals_between_peak_cluster == True only align peak cluster if False align all intervals
        # TODO: still a bit experimental
        idx, signal = get_index_signal_with_highest_correlation(self._signals)
        peaks, _ = find_peaks(signal,
                              prominence=np.max(signal) * self._align_auto_pick_min_peak_height_factor,
                              threshold=np.max(signal) * self._align_auto_pick_min_peak_height_factor * 0.04)
        #                      height=int(np.max(signal) * self._align_auto_pick_min_peak_height_factor))
        non_adjacent = False
        split_list = []
        ignore_intervals_no = []
        last_peak = 0
        peak = 0
        offset = int(round(self._signal_length * self._align_auto_pick_offset_factor))
        in_cluster_max_dist = int(round(self._signal_length * self._align_auto_pick_max_cluster_dist_factor))
        i = 0
        if non_adjacent:
            for peak in peaks:
                if last_peak == 0:
                    split_list.append(peak - offset)
                    ignore_intervals_no.append(i)
                    i += 1
                    last_peak = peak
                if peak > last_peak + in_cluster_max_dist:
                    split_list.append(last_peak + offset)
                    i += 1
                    ignore_intervals_no.append(i)
                    split_list.append(peak - offset)
                    i += 1
                last_peak = peak
            split_list.append(peak + offset)
            i += 1
            ignore_intervals_no.append(i)
            if self._align_auto_pick_ignore_intervals_between_peak_cluster:
                self._ignore_intervals_no = ignore_intervals_no
        else:
            for peak in peaks:
                if last_peak == 0:
                    split_list.append(peak-offset)
                    last_peak = peak
                if peak > last_peak + in_cluster_max_dist:
                    split_list.append(int((last_peak+peak)/2))
                last_peak = peak
            split_list.append(peak + offset)
        return split_list

    # max shift correction functions

    def _max_shift_manual(self, target, signals):
        shifts, scores = ccfftshift(target, signals,
                                    -self._max_shift_correction, self._max_shift_correction)
        return shifts, scores, self._max_shift_correction

    def _max_shift_best(self, target, signals):
        return self._find_shift_loop(target, signals, self._best_shift_find_increment)

    def _max_shift_fast(self, target, signals):
        return self._find_shift_loop(target, signals, self._fast_shift_find_increment)

    def _find_shift_loop(self, target, signals, shift_increment):
        n = 1
        shifts = None
        scores = None
        for i in range(0, self._max_shift_find_iterations):
            shifts, scores = ccfftshift(target, signals,
                                        -n, n)
            if np.max(shifts) < n:
                logging.info('Found optimal max shift %i in iteration %i using mode %s' % (
                    n, i, self._max_shift_correction_mode))
                break
            if i == self._max_shift_find_iterations:
                logging.warning('Reached max interations but did not find optimal max shift')
            n = n + shift_increment
        return shifts, scores, n

    def _convert_to_datapts(self, value):
        # returns the index of the datapoint closest to the input value
        if np.abs(self._unit_to_sample_point_map - value) != 0:
            logging.warning('Could not convert %f exactly to datapoint - using closest datapoint' % value)
        return np.argmin(np.abs(self._unit_to_sample_point_map - value))

    def _convert_length_to_datapts(self, value):
        factor = 1 / abs(self._unit_to_sample_point_map[0] - self._unit_to_sample_point_map[1])
        if value % (1 / factor) != 0:
            logging.warning('Could not convert %f exactly to datapoints - round to closest datapoints' % value)
        return int(round(factor * value))

    def test_data(self, use_set=1):
        if use_set == 1:
            self._signals = np.array([
                [0, 0, 0, 2, 3, 2, 0, 0, 2, 1, 2, np.nan, 3, 4, 2, 3, 2, 0, 2, 3, 2],
                [0, 0, 2, 3, 2, 0, 0, 2, 3, 1, 2, np.nan, 3, 4, 3, 2, 0, 2, 3, 2, 0],
                [0, 2, 3, 2, 0, 0, 2, 3, 2, 1, 2, np.nan, 3, 4, 2, 0, 2, 3, 2, 0, 1],
            ]).astype(np.float)
            self._signal_length = self._signals[0].shape[0]
        elif use_set == 2:
            self._signals = np.array([[np.nan, 0, 0, 0, 0, 0, 0, 1, 2, 4, 8, 4, 2, 1, 0, 0],
                                      [np.nan, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 4, 8, 4, 2, 1],
                                      [0, 1, 2, 4, 8, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, np.nan]]).astype(np.float)

            self._target = np.array([0, 0, 0, 0, 0, 1, 2, 4, 8, 4, 2, 1, 0, 0, 0, 0]).astype(np.float)
            self._signal_length = self._signals[0].shape[0]
