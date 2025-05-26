import numpy as np


class normalizer():
    def __init__(self, max=None, min=None, mean=None, std=None):
        self.max = max
        self.min = min
        self.mean = mean
        self.std = std

    @staticmethod
    def IQ2AP(data):
        # data format: [Num_Frames, I/Q, Num_Dopplers, Num_Angles, Num_Ranges]
        amp_phase_data = np.zeros(data.shape)
        amp_phase_data[:, 0, ...] = np.sqrt(data[:, 0, ...] ** 2 + data[:, 1, ...] ** 2)
        amp_phase_data[:, 1, ...] = np.arctan2(data[:, 1, ...], data[:, 0, ...])
        return amp_phase_data

    @staticmethod
    def IQ2AP_Unwrap(data):
        # data format: [Num_Frames, I/Q, Num_Dopplers, Num_Angles, Num_Ranges]
        amp_phase_data = np.zeros(data.shape)
        amp_phase_data[:, 0, ...] = np.sqrt(data[:, 0, ...] ** 2 + data[:, 1, ...] ** 2)
        amp_phase_data[:, 1, ...] = np.unwrap(np.arctan2(data[:, 1, ...], data[:, 0, ...]), axis=1)
        return amp_phase_data

    @staticmethod
    def AP2IQ(data):
        # data format: [Num_Frames, 2, Num_Angles, Num_Ranges] 2: Amplitude/Phase
        IQ_data = np.zeros(data.shape)
        complex_data = data[:, 0, ...] * np.exp(1j * data[:, 1, ...])
        IQ_data[:, 0, ...] = complex_data.real
        IQ_data[:, 1, ...] = complex_data.imag
        return IQ_data

    def normalize_max(self, data):
        # for the test_set, use the min_max value of the train_set
        # first turn the data into energy-phase domain, then divide the max energy
        # finally turn back to I/Q domain
        # data format: [2, fast time, slow time, #samples]
        return data / self.max

    def normalize_min_max(self, data):
        return (data - self.min) / (self.max - self.min)

    def normalize_mean_std_amplitude(self, data):
        ### data: [Num_Frames, I/Q,  Num_Dopplers, Num_Angles, Num_Ranges]
        ap = self.IQ2AP(data)
        ap[:, 0, ...] = (ap[:, 0, ...] - self.mean) / self.std
        iq = self.AP2IQ(ap)
        return iq
