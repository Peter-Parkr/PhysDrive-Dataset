import os
import numpy as np
import scipy.io as sio
from radar_util import hampel
import pandas as pd
from scipy.interpolate import interp1d
import normalizer
import matplotlib.pyplot as plt

NEAREST_RANGE_BIN = 4  # 4 * 0.07 m = 0.28 m


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
    # plt.imshow(range_doppler_heatmap)
    ### 1. Hampel filter
    range_results_pd = np.array(range_results)
    ### 2. Displacement threshold
    range_displacements = np.abs(range_results_pd[1:] - range_results_pd[:-1])
    outlier_idx = np.zeros(range_results_pd.shape[0], dtype=bool)
    # outlier_idx[0:range_displacements.shape[0]] = np.abs(range_displacements) > 3 # 3 * 0.07 m = 0.21 m, 0.21 m / 0.05s = 4.2 m/s as the displacement threshold
    outlier_idx[1:] = np.abs(range_displacements) >= 3
    x = np.arange(0, range_results_pd.shape[0])
    f = interp1d(x[~outlier_idx], range_results_pd[~outlier_idx], kind='linear', fill_value='extrapolate')
    interp_values = f(outlier_idx.nonzero()[0])
    filtered_range_results = np.zeros(range_results_pd.shape[0])
    filtered_range_results[outlier_idx] = interp_values
    filtered_range_results[~outlier_idx] = range_results_pd[~outlier_idx]
    filtered_range_results = filtered_range_results.astype(int)

    # range_displacements = np.abs(range_results_pd[2:] - range_results_pd[:-2])
    range_displacements = np.abs(filtered_range_results[2:] - filtered_range_results[:-2])
    outlier_idx = np.zeros(range_results_pd.shape[0], dtype=bool)
    # outlier_idx[0:range_displacements.shape[0]] = np.abs(range_displacements) > 3 # 6 * 0.07 m = 0.42 m, 0.42 m / 0.10s = 4.2 m/s as the displacement threshold
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


global_sum = 0
global_num = 0

if __name__ == "__main__":
    rootdir = ''  # Change it to your folder address
    subs = os.listdir(rootdir)
    subs = sorted(subs)
    subs = [sub for sub in subs if 'DS_' not in sub and 'cfg' not in sub]
    for sub in subs:
        print('Now Processing:', sub)
        sub_path = os.path.join(rootdir, sub)
        output = os.path.join(sub_path, 'processed')
        files = os.listdir(output)
        files = [file for file in files if '.mat' in file]
        files = sorted(files)
        for file in files:
            file_path = os.path.join(output, file)
            print('Now Processing:', file_path)
            data = sio.loadmat(file_path)
            range_azimuth_doppler_fft = data['range_azimuth_doppler_fft']

            frame_ids, range_results, angle_results = select_regions_by_heatmaps(range_azimuth_doppler_fft)
            cropped_range_azimuth_doppler_fft = crop_heatmaps(range_azimuth_doppler_fft, frame_ids, range_results,
                                                              angle_results)
            global_sum += np.abs(cropped_range_azimuth_doppler_fft).sum()
            global_num += cropped_range_azimuth_doppler_fft.size

            sio.savemat(
                file_path,
                {'range_azimuth_doppler_fft': cropped_range_azimuth_doppler_fft})
            print('After cropping range_azimuth_doppler_fft shape:', cropped_range_azimuth_doppler_fft.shape)


    print('Start normalization and I/Q seperation...')
    global_mean = global_sum / global_num
    global_std = 0
    for sub in subs:
        sub_path = os.path.join(rootdir, sub)
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
        sub_path = os.path.join(rootdir, sub)
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
