import os
import numpy as np
from file_parse import parse, organize
from radar_util import read_config, TI_RADAR_TYPE
from radar_processing import rangeFFT, dopplerFFT, angleFFT, clutter_removal
from dsp_util import Window
import matplotlib.pyplot as plt
import scipy.io as sio
from object_detection import cfar2d
from radar_util import hampel
import pandas as pd
from scipy.interpolate import interp1d
import normalizer
from glob import glob
import re
from tqdm import tqdm



NEAREST_RANGE_BIN = 4 # 4 * 0.07 m = 0.28 m

def calculate_resolutions(radar_config):
    """Calculate radar resolutions"""
    # Range resolution
    c = 3e8
    effective_bandwidth = radar_config['freq_slope_hz_sec'] * radar_config['n_sample'] / (
                0.9 * radar_config['ADC_rate'])

    max_range = 0.9 * radar_config['ADC_rate'] * c / (2 * radar_config['freq_slope_hz_sec'])
    range_res = c / (2 * effective_bandwidth)

    wavelength = c / 77e9

    chirp_period = radar_config['n_tx'] * (radar_config['idle_time_sec'] +
                                           radar_config['ramp_end_time_sec'])

    max_vel = wavelength / (4 * chirp_period)
    vel_res = (2 * max_vel) / radar_config['n_loop']

    return max_range, range_res, max_vel, vel_res


def plot_range_angle_doppler_profile(range_azimuth_doppler_fft, frame_id, radar_config, range_res, max_vel):
    """Plot range-angle-doppler profile"""
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Range-Doppler plot
    distances = np.linspace(0, range_res * radar_config['n_sample'], radar_config['n_sample'])[
                :range_azimuth_doppler_fft.shape[3]]
    velocities = np.linspace(-max_vel, max_vel, radar_config['n_loop'])[
                 (radar_config['n_loop'] - range_azimuth_doppler_fft.shape[1]) // 2:
                 (radar_config['n_loop'] + range_azimuth_doppler_fft.shape[1]) // 2]

    im1 = ax1.pcolormesh(distances, velocities,
                         np.abs(range_azimuth_doppler_fft[frame_id].sum(axis=1)),
                         shading='nearest')
    fig.colorbar(im1, ax=ax1, label='Magnitude (dB)')
    ax1.set_title('Range-Doppler Map')
    ax1.set_xlabel('Range (m)')
    ax1.set_ylabel('Velocity (m/s)')

    # Range-Angle plot
    angles = np.linspace(1, -1, range_azimuth_doppler_fft.shape[2])
    angles = np.rad2deg(np.arccos(angles))
    ax2.pcolormesh(distances, angles, np.abs(range_azimuth_doppler_fft[frame_id].sum(axis=0)))
    ax2.set_title('Range-Angle Map')
    ax2.set_xlabel('Range (m)')
    ax2.set_ylabel('Angle (degrees)')

    # Range profile
    ax3.plot(distances,
             np.abs(range_azimuth_doppler_fft[frame_id, range_azimuth_doppler_fft.shape[1] // 2, :, :].sum(axis=0)))
    ax3.set_title('Range Profile')
    ax3.set_xlabel('Range (m)')
    ax3.set_ylabel('Magnitude')

    plt.tight_layout()
    plt.show()


def process_radar_data(file_name, config_file,
                       range_angle_doppler_limits=[32, None, 64]):
    """Process radar data with given configuration"""
    # Read radar configuration
    radar_config = read_config(config_file=config_file,
                               layout=TI_RADAR_TYPE.iwr1843boost)

    # Read radar data
    adc_data = np.fromfile(file_name, dtype=np.int16)

    # Parse and organize radar data
    adc_frames = parse(adc_data=adc_data, radar_config=radar_config)
    adc_frames = organize(adc_frames=adc_frames,
                          layout=TI_RADAR_TYPE.iwr1843boost,
                          chirp_cfg=radar_config['chirp_cfg'],
                          n_rx=radar_config['n_rx'])

    # Range FFT
    range_fft_data = rangeFFT(adc_frames,
                              n_range_fft=radar_config['n_sample'],
                              window_type_1d=Window.HANNING, axis=-1)
    ###
    range_fft_data, _ = clutter_removal(range_fft_data, axis=1)

    range_fft_data = range_fft_data[..., :range_angle_doppler_limits[0]]  # limits to 32 * 0.07 m = 2.24 m

    # Doppler FFT
    range_doppler_fft = dopplerFFT(range_fft_data,
                                   num_chirps=radar_config['n_loop'],
                                   axis=1)
    range_doppler_fft = range_doppler_fft[:, radar_config['n_loop'] // 2 - range_angle_doppler_limits[1] // 2:
                                             radar_config['n_loop'] // 2 + range_angle_doppler_limits[1] // 2, :,
                        :]  # limits to +- 32*0.06 m/s = +- 1.92 m/s

    # Angle FFT (for azimuth antennas)
    # no limits on angle
    range_azimuth_doppler_fft = angleFFT(range_doppler_fft, 'azimuth', FFT_size=32, numRx=4, axis=2)
    print('After cropping range_azimuth_doppler_fft shape:', range_azimuth_doppler_fft.shape)

    return range_azimuth_doppler_fft


def select_regions_by_heatmaps(range_azimuth_doppler_fft, debug=False):
    """Select regions by CFAR"""
    # Calculate noise power
    ### range_azimuth_doppler_fft: [n_frames, n_doppler, n_ant, n_range], complex-valued
    ### return: selected regions

    selected_locations = []
    frame_ids = []
    range_results = []

    for frame_id in range(range_azimuth_doppler_fft.shape[0]):
        range_azimuth_doppler_heatmap = np.abs(range_azimuth_doppler_fft[frame_id]) ** 2

        range_doppler_heatmap = range_azimuth_doppler_heatmap.sum(axis=1)
        if debug:
            if frame_id % 100 == 0:
                plt.figure()
                plt.title(f'Range-Doppler Heatmap [frame_id]: {frame_id}')
                plt.imshow(range_doppler_heatmap)
        n_doppler = range_doppler_heatmap.shape[0]

        ### we mainly care about the energy from the doppler bins within [-2*0.06, 2*0.06] m/s,
        ### we only consider the range bins at least NEAREST_RANGE_BIN away, i.e., 0.07 * 4 = 0.28 m
        range_result = range_doppler_heatmap[n_doppler // 2 - 2:n_doppler // 2 + 3, NEAREST_RANGE_BIN:].sum(
            axis=0).argmax() + NEAREST_RANGE_BIN

        frame_ids.append(frame_id)
        range_results.append(range_result)

    ### 1. Hampel filter
    range_results_pd = np.array(range_results)
    ### 2. Displacement threshold
    range_displacements = np.abs(range_results_pd[1:] - range_results_pd[:-1])
    outlier_idx = np.zeros(range_results_pd.shape[0], dtype=bool)
    outlier_idx[1:] = np.abs(range_displacements) >= 3
    x = np.arange(0, range_results_pd.shape[0])
    f = interp1d(x[~outlier_idx], range_results_pd[~outlier_idx], kind='linear', fill_value='extrapolate')
    interp_values = f(outlier_idx.nonzero()[0])
    filtered_range_results = np.zeros(range_results_pd.shape[0])
    filtered_range_results[outlier_idx] = interp_values
    filtered_range_results[~outlier_idx] = range_results_pd[~outlier_idx]
    filtered_range_results = filtered_range_results.astype(int)

    range_displacements = np.abs(filtered_range_results[2:] - filtered_range_results[:-2])
    outlier_idx = np.zeros(range_results_pd.shape[0], dtype=bool)

    outlier_idx[2:] = np.abs(range_displacements) >= 6
    x = np.arange(0, range_results_pd.shape[0])
    f = interp1d(x[~outlier_idx], filtered_range_results[~outlier_idx], kind='linear', fill_value='extrapolate')
    interp_values = f(outlier_idx.nonzero()[0])
    filtered_range_results_2 = np.zeros(range_results_pd.shape[0])
    filtered_range_results_2[outlier_idx] = interp_values
    filtered_range_results_2[~outlier_idx] = filtered_range_results[~outlier_idx]
    filtered_range_results_2 = filtered_range_results_2.astype(int)

    # angle results
    angle_results = []
    for i in range(len(frame_ids)):
        range_azimuth_doppler_heatmap = np.abs(range_azimuth_doppler_fft[frame_id]) ** 2

        range_azimuth_heatmap = range_azimuth_doppler_heatmap.sum(axis=0)  # [n_angle, n_range]
        range_result = filtered_range_results_2[i]
        angle_result = range_azimuth_heatmap[:,
                       max(range_result - 4, 0): min(range_result + 4, range_azimuth_heatmap.shape[1] - 1)].sum(
            -1).argmax()
        angle_results.append(angle_result)

    angle_results_pd = pd.DataFrame(angle_results)
    angle_results_pd, _ = hampel(angle_results_pd, k=5, t0=3)
    angle_results_pd = angle_results_pd.squeeze()

    selected_locations = np.array(frame_ids), np.array(filtered_range_results), np.array(angle_results)
    if debug:
        plt.figure()
        plt.plot(range_results, label='range_results')
        plt.plot(range_results_pd, label='range_results_pd')
        plt.plot(filtered_range_results, label='filtered_range_results')
        plt.legend()
        plt.show()
        plt.figure()
        plt.plot(angle_results, label='angle_results')
        plt.plot(angle_results_pd, label='angle_results_pd')
        plt.legend()
        plt.show()

    return selected_locations


def crop_heatmaps(range_azimuth_doppler_fft, frame_ids, range_results, angle_results,
                  range_angle_doppler_limits=[8, 16, 8]):
    """Crop the heatmaps"""
    ### range_azimuth_doppler_fft: [n_frames, n_doppler, n_ant, n_range], complex-valued
    ### frame_ids: [n_frames]
    ### range_results: [n_frames]
    ### angle_results: [n_frames]
    ### return: cropped range_azimuth_doppler_fft

    cropped_range_azimuth_doppler_fft = np.zeros(
        (len(frame_ids), range_angle_doppler_limits[2], range_angle_doppler_limits[1], range_angle_doppler_limits[0]),
        dtype=np.complex64)
    for frame_id, range_result, angle_result in zip(frame_ids, range_results, angle_results):
        range_azimuth_doppler_sample = range_azimuth_doppler_fft[frame_id]
        n_doppler = range_azimuth_doppler_sample.shape[0]
        if range_result - range_angle_doppler_limits[0] // 2 < NEAREST_RANGE_BIN:
            ### we only consider the range bins at least NEAREST_RANGE_BIN away, i.e., 0.07 * 4 = 0.28 m
            range_start = NEAREST_RANGE_BIN
            range_end = range_angle_doppler_limits[0] + NEAREST_RANGE_BIN
        elif range_result + range_angle_doppler_limits[0] // 2 > range_azimuth_doppler_sample.shape[2]:
            range_start = range_azimuth_doppler_sample.shape[2] - range_angle_doppler_limits[0]
            range_end = range_azimuth_doppler_sample.shape[2]
        else:
            range_start = range_result - range_angle_doppler_limits[0] // 2
            range_end = range_result + range_angle_doppler_limits[0] // 2

        if angle_result - range_angle_doppler_limits[1] // 2 < 0:
            angle_start = 0
            angle_end = range_angle_doppler_limits[1]
        elif angle_result + range_angle_doppler_limits[1] // 2 > range_azimuth_doppler_sample.shape[1]:
            angle_start = range_azimuth_doppler_sample.shape[1] - range_angle_doppler_limits[1]
            angle_end = range_azimuth_doppler_sample.shape[1]
        else:
            angle_start = angle_result - range_angle_doppler_limits[1] // 2
            angle_end = angle_result + range_angle_doppler_limits[1] // 2

        cropped_sample = range_azimuth_doppler_sample[
                         n_doppler // 2 - range_angle_doppler_limits[2] // 2:n_doppler // 2 +
                                                                             range_angle_doppler_limits[2] // 2,
                         angle_start:angle_end,
                         range_start:range_end]
        cropped_range_azimuth_doppler_fft[frame_id] = cropped_sample

    return cropped_range_azimuth_doppler_fft


def extract_number(filename):
    """ Extracting numbers from file names """
    match = re.search(r'adc_data_Raw_(\d+)\.bin', filename)
    return int(match.group(1)) if match else float('inf')  # If it can't match, put it to the end

def sort_bin_files(directory):
    """ Sort by number in filename """
    bin_files = glob(os.path.join(directory, 'adc_data_Raw_*.bin'))  # Get all .bin files
    sorted_files = sorted(bin_files, key=extract_number)  # Sort by number
    return sorted_files

def main():
    global_sum = 0
    global_num = 0
    rootdir = '' # Change it to your folder address
    subs = os.listdir(rootdir)
    subs = sorted(subs)
    subs = [sub for sub in subs if 'DS_' not in sub and 'cfg' not in sub]
    config_file = 'iwr1843boost.cfg'
    for sub in subs:
        print('Now Processing:', sub)
        sub_path = os.path.join(os.path.join(rootdir, sub), 'mmWave')
        files = sort_bin_files(sub_path)

        output = os.path.join(sub_path, 'processed')
        if not os.path.exists(output):
            os.makedirs(output)

        for data_path in tqdm(files):
            file = data_path.split('/')[-1]
            idx = file.split('_')[-1].split('.')[0]
            range_azimuth_doppler_fft = process_radar_data(data_path, config_file,
                                                           range_angle_doppler_limits=[32, 32, 64])
            frame_ids, range_results, angle_results = select_regions_by_heatmaps(range_azimuth_doppler_fft)
            cropped_range_azimuth_doppler_fft = crop_heatmaps(range_azimuth_doppler_fft, frame_ids, range_results,
                                                              angle_results)

            global_sum += np.abs(cropped_range_azimuth_doppler_fft).sum()
            global_num += cropped_range_azimuth_doppler_fft.size

            print('After cropping range_azimuth_doppler_fft shape:', cropped_range_azimuth_doppler_fft.shape)
            sio.savemat(
                os.path.join(output, 'range_azimuth_doppler_fft_' + idx + '.mat'),
                {'range_azimuth_doppler_fft': cropped_range_azimuth_doppler_fft})

    print('Start normalization and I/Q seperation...')
    global_mean = global_sum / global_num
    global_std = 0
    for sub in subs:
        sub_path = os.path.join(os.path.join(rootdir, sub), 'mmWave')
        output = os.path.join(sub_path, 'processed')
        files = os.listdir(output)
        files = [file for file in files if '.mat' in file]
        files = sorted(files)
        for file in files:
            file_path = os.path.join(output, file)
            data = sio.loadmat(file_path)
            range_azimuth_doppler_fft = data['range_azimuth_doppler_fft']
            global_std += ((np.abs(range_azimuth_doppler_fft) - global_mean) ** 2).sum()

            range_azimuth_doppler_fft = np.stack((range_azimuth_doppler_fft.real, range_azimuth_doppler_fft.imag),
                                                 axis=1)

            sio.savemat(
                file_path,
                {'range_azimuth_doppler_fft': range_azimuth_doppler_fft})
    global_std = np.sqrt(global_std / global_num)
    my_normalizer = normalizer.normalizer(mean=global_mean, std=global_std)
    for sub in subs:
        sub_path = os.path.join(os.path.join(rootdir, sub), 'mmWave')
        output = os.path.join(sub_path, 'processed')
        files = os.listdir(output)
        files = [file for file in files if '.mat' in file]
        files = sorted(files)
        for file in files:
            file_path = os.path.join(output, file)
            data = sio.loadmat(file_path)
            range_azimuth_doppler_fft = data['range_azimuth_doppler_fft']
            range_azimuth_doppler_fft = my_normalizer.normalize_mean_std_amplitude(range_azimuth_doppler_fft)
            sio.savemat(
                file_path,
                {'range_azimuth_doppler_fft': range_azimuth_doppler_fft})


if __name__ == "__main__":
    main()
