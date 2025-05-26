import numpy as np
from dsp_util import windowing

def rangeFFT(adc_data, n_range_fft=None, window_type_1d=None, axis=-1):
    """Perform 1D FFT on complex-format ADC data.

    Perform optional windowing and 1D FFT on the ADC data.

    Args:
        adc_data (ndarray): (n_loop, n_virtual_ant, n_sample). Performed on each frame. adc_data
                            is in complex by default. Complex is float32/float32 by default.
        window_type_1d (dsp_utils.Window): Optional window type on 1D FFT input. Default is None. Can be selected
                                                from Bartlett, Blackman, Hanning and Hamming.
    
    Returns:
        radar_cube (ndarray): (n_loop, n_virtual_ant, num_range_bins). Also called fft_1d_out
    """
    if n_range_fft is None:
        n_range_fft = adc_data.shape[axis]

    # Note: np.fft.fft is a 1D operation, using higher dimension input defaults to slicing last axis for transformation
    # windowing numA x numB suggests the coefficients is numA-bits while the 
    # input and output are numB-bits. Same rule applies to the FFT.
    fft1d_window_type = window_type_1d
    if fft1d_window_type:
        fft1d_in = windowing(adc_data, fft1d_window_type, axis=axis)
    else:
        fft1d_in = adc_data

    # Note: np.fft.fft is a 1D operation, using higher dimension input defaults to slicing last axis for transformation
    # radar_cube = np.fft.fftshift(np.fft.fft(fft1d_in, n=n_range_fft, axis=axis), axes=axis)
    radar_cube = np.fft.fft(fft1d_in, n=n_range_fft, axis=axis)

    return radar_cube

def dopplerFFT(rangeResult, num_chirps, axis=1):
      # [n_frames, n_doppler, n_ant, n_range]
	windowedBins2D = rangeResult * \
		np.reshape(np.hamming(num_chirps), (1, -1, 1, 1))
	dopplerFFTResult = np.fft.fft(windowedBins2D, axis=axis)
	dopplerFFTResult = np.fft.fftshift(dopplerFFTResult, axes=axis)
	return dopplerFFTResult

def angleFFT(rangeResult, angle, FFT_size, axis=2, numRx=4):

    # range_fft / range_doppler_fft: [n_frames, n_doppler, n_ant, n_range]
	n_frames, n_doppler, n_ant, n_range = rangeResult.shape

	angle_padded = np.zeros(shape=(n_frames, n_doppler, FFT_size, n_range), dtype=np.complex_)
	if angle == 'azimuth':
		angle_padded[:, :, :2*numRx, :] = rangeResult[:, :,  :2*numRx, :]
	elif angle == 'elevation':
		angle_padded[:, :, int(n_ant-2*numRx):, :] = rangeResult[:, :, 2*numRx:, :]
	angle_fft = np.fft.fft(angle_padded, axis=axis)
	angle_fft = np.fft.fftshift(angle_fft, axes=axis)
	return angle_fft

def clutter_removal(input_val, axis=0):
    """Perform basic static clutter removal by removing the mean from the input_val on the specified doppler axis.

    Args:
        input_val (ndarray): Array to perform static clutter removal on. Usually applied before performing doppler FFT.
            e.g. [n_loop, n_virtual_ant, n_range_bin], it is applied along the first axis.
        axis (int): Axis to calculate mean of pre-doppler.

    Returns:
        ndarray: Array with static clutter removed.
        mean: Array with static clutter
    """
    # Reorder the axes
    reordering = np.arange(len(input_val.shape))
    reordering[0] = axis
    reordering[axis] = 0
    input_val = input_val.transpose(reordering)

    # Apply static clutter removal
    mean = input_val.mean(0)
    output_val = input_val - mean

    return output_val.transpose(reordering), mean