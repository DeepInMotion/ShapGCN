from abc import ABC, abstractmethod
import numpy as np
from scipy import signal
import logging


class BaseDataLoader(ABC):
    def __init__(self):
        pass

    @staticmethod
    def butterworth_filter(data, butterworth_coeffs):
        """
        Filter the signal with the given filter order.
        :param data: Input data to be filtered, shape (C, T, V, M)
        :param butterworth_coeffs: Filter coefficients
        :return: Filtered signal
        """
        C, T, V, M = data.shape
        filtered_signal = np.zeros((C, T, V, M))
        for m in range(M):
            for v in range(V):
                for c in range(C):
                    filtered_signal[c, :, v, m] = signal.sosfiltfilt(butterworth_coeffs, data[c, :, v, m])
        return filtered_signal

    def mean_std(self, data):
        """
        C == channels,
        M == skeletons,
        N == number of skeletons,
        V == vertices,
        T == frame
        :return:
        """
        N, C, T, V = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=0, keepdims=True) # .mean(axis=4, keepdims=True)
        self.std_map = data.transpose((0, 2, 1, 3)).reshape((N * T, C * V)).std(axis=0).reshape((C, 1, V))
        logging.info('Mean value: {} \n Std value: {}'.format(self.mean_map, self.std_map))
