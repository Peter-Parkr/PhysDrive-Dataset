import os
import numpy as np
import pandas as pd
import scipy.io as sio
from datetime import datetime, timezone, timedelta
import neurokit2 as nk
from biosppy.signals import bvp
import mne
from tqdm import tqdm
from glob import glob
import re
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def extract_number(filename):
    """ Extracting numbers from file names """
    match = re.search(r'range_azimuth_doppler_fft_(\d+)\.mat', filename)
    return int(match.group(1)) if match else float('inf')  # If it can't match, put it to the end


def sort_bin_files(directory):
    """ Sort by number in filename """
    bin_files = glob(os.path.join(directory, 'range_azimuth_doppler_fft_*.mat'))  # Get all .bin files
    sorted_files = sorted(bin_files, key=extract_number)  # Sort by number
    return sorted_files


rootdir = '' # Change it to your folder address
outputdir = os.path.join(rootdir, 'processed')
subs = os.listdir(rootdir)
subs = sorted(subs)

subs = [sub for sub in subs if 'DS_' not in sub]
subs = [sub for sub in subs if not sub.startswith('.')]
print(subs)

sampling_rate_mmwave = 20  # Millimeter-wave radar sampling rate (Hz)
frames_per_mat = 1365  # Standard number of frames per .mat file

for sub in tqdm(subs):

    phys_dir = os.path.join(rootdir, sub, 'HRT')
    mmwave_dir = os.path.join(rootdir, sub, 'mmWave')

    # -------------- mmWave Data --------------
    # Read start and end times from log file
    temp = pd.read_csv(os.path.join(mmwave_dir, 'adc_data_Raw_LogFile.csv'), usecols=[0])
    mmwave_start_time_str = temp.values[-3][0]
    mmwave_end_time_str = temp.values[-2][0]
    mmwave_start_time = datetime.strptime(mmwave_start_time_str.split(' - ')[1],
                                          '%a %b %d %H:%M:%S %Y')
    mmwave_end_time = datetime.strptime(mmwave_end_time_str.split(' - ')[1],
                                        '%a %b %d %H:%M:%S %Y')
    mmwave_processed = os.path.join(mmwave_dir, 'processed')
    mmwave_files = sort_bin_files(mmwave_processed)
    mmwave_files = [os.path.basename(path) for path in mmwave_files]
    mmwave_timestamps = []
    frame_counter = 0

    for idx, mm_file in enumerate(mmwave_files):
        mat_data = sio.loadmat(os.path.join(mmwave_processed, mm_file))
        num_frames = mat_data['range_azimuth_doppler_fft'].shape[0]
        start_time = mmwave_start_time + timedelta(seconds=frame_counter / sampling_rate_mmwave)
        end_time = start_time + timedelta(seconds=num_frames / sampling_rate_mmwave)
        mmwave_timestamps.append((start_time, end_time, num_frames, mm_file))
        frame_counter += num_frames

    # -------------- Physiological Data (ECG/RESP) --------------
    for idx, file_id in enumerate(sorted([s for s in os.listdir(phys_dir) if 'DS_' not in s])):
        # temp_seq = os.path.join(phys_dir, file_id)
        temp_seq = phys_dir
        file_id = [s for s in os.listdir(temp_seq) if s.endswith('.csv')]
        data = pd.read_csv(os.path.join(temp_seq, file_id[0]), encoding='ISO-8859-1')
        cols = [col for col in data.columns if 'ECG' in col or 'RESP' in col or 'StorageTime' in col]
        data = data[cols].dropna()

        # Get timestamps and compute sampling frequency
        phys_start_timestamp = data.iloc[0, 0] #/ 1e7
        phys_end_timestamp = data.iloc[-1, 0] #/ 1e7
        phys_start_dt = datetime.strptime(phys_start_timestamp, '%Y-%m-%d %H:%M:%S.%f')
        phys_end_dt = datetime.strptime(phys_end_timestamp, '%Y-%m-%d %H:%M:%S.%f')
        diff_seconds = (phys_end_dt - phys_start_dt).total_seconds()
        phys_sampling_freq = data.shape[0] / diff_seconds

        data.index = list(range(data.shape[0]))

        ecg_series = pd.to_numeric(data.iloc[:, 1], errors='coerce').astype(float)
        rsp_series = pd.to_numeric(data.iloc[:, 2], errors='coerce').astype(float)
        data = data.values

        # Clean the ECG and RESP signals

        data[:, 1] = nk.ecg_clean(ecg_series, sampling_rate=int(phys_sampling_freq))
        data[:, 2] = nk.rsp_clean(rsp_series, sampling_rate=int(phys_sampling_freq))

        # Convert to a common timezone (UTC+8)
        phys_start_time = phys_start_dt
        phys_end_time = phys_end_dt

        # Create a downsampled time range for physiological data (target 20 Hz)
        phys_time_range = []
        phys_downsampled_time_range = []
        phys_downsampled_signal = []
        current_time = phys_start_time
        while current_time < phys_end_time:
            phys_time_range.append(current_time)
            current_time += timedelta(seconds=1 / int(phys_sampling_freq))
        for i in range(0, len(phys_time_range), int(phys_sampling_freq / 20)):
            phys_downsampled_time_range.append(phys_time_range[i])
            if i < len(data):
                phys_downsampled_signal.append(data[i])

        # -------------- Determine the Shared Segment --------------
        # Common segment is the intersection of mmWave, physiological, and EDF time ranges.
        shared_start_time = max(mmwave_start_time, phys_start_time)
        shared_end_time = min(mmwave_end_time, phys_end_time)
        if (shared_end_time - shared_start_time) < timedelta(seconds=20):
            continue

        # -------------- Filter Physiological Data to Shared Segment --------------
        final_phys_time, final_ECG, final_RESP = [], [], []
        for time, value in zip(phys_downsampled_time_range, phys_downsampled_signal):
            if shared_start_time <= time <= shared_end_time:
                final_phys_time.append(time)
                final_ECG.append(value[1])
                final_RESP.append(value[2])

        # -------------- mmWave Data Matching --------------
        matched_mmwave_data = []
        for start_time, end_time, num_frames, file_name in mmwave_timestamps:
            if end_time < shared_start_time or start_time > shared_end_time:
                continue
            mat_data = sio.loadmat(os.path.join(mmwave_processed, file_name))['range_azimuth_doppler_fft']
            start_index = max(0, int((shared_start_time - start_time).total_seconds() * sampling_rate_mmwave))
            end_index = min(num_frames, int((shared_end_time - start_time).total_seconds() * sampling_rate_mmwave))
            matched_mmwave_data.append(mat_data[start_index:end_index])
        if matched_mmwave_data:
            matched_mmwave_data = np.concatenate(matched_mmwave_data, axis=0)

        # Calculate the number of samples for each modality (at 20 Hz)
        n_phys = len(final_ECG)
        n_mmwave = matched_mmwave_data.shape[0] if matched_mmwave_data.size > 0 else 0

        common_length = min(n_phys, n_mmwave)
        if common_length == 0:
            print(f"Insufficient overlapping data for subject {sub} file {file_id}")
            continue

        # Trim each modality to the common_length
        final_ECG = final_ECG[:common_length]
        final_RESP = final_RESP[:common_length]
        if matched_mmwave_data.size > 0:
            matched_mmwave_data = matched_mmwave_data[:common_length]

        # -------------- Save the Synchronized Data --------------
        if len(final_ECG) / 20 > 30:
            for i in range(len(final_ECG) // 600):
                output = os.path.join(outputdir, sub + '_' + str(idx) + str(i))
                if not os.path.exists(output):
                    os.makedirs(output)
                sio.savemat(os.path.join(output, 'ecg.mat'), {'ecg': final_ECG[i * 600:(i + 1) * 600]})
                sio.savemat(os.path.join(output, 'resp.mat'), {'resp': final_RESP[i * 600:(i + 1) * 600]})

                if len(matched_mmwave_data) > 0:
                    sio.savemat(os.path.join(output, 'mmwave.mat'),
                                {'mmwave': matched_mmwave_data[i * 600:(i + 1) * 600]})
                print("Physio duration (s):", len(final_ECG[i * 600:(i + 1) * 600]) / 20)
                print("mmWave duration (s):", len(matched_mmwave_data[i * 600:(i + 1) * 600]) / 20)
                print()
            if len(final_ECG) - (i + 1) * 600 > 220:
                output = os.path.join(outputdir, sub + '_' + str(idx) + str(i + 1))
                if not os.path.exists(output):
                    os.makedirs(output)
                sio.savemat(os.path.join(output, 'ecg.mat'), {'ecg': final_ECG[(i + 1) * 600:]})
                sio.savemat(os.path.join(output, 'resp.mat'), {'resp': final_RESP[(i + 1) * 600:]})

                if len(matched_mmwave_data) > 0:
                    sio.savemat(os.path.join(output, 'mmwave.mat'), {'mmwave': matched_mmwave_data[(i + 1) * 600:]})
                print("Physio duration (s):", len(final_ECG[(i + 1) * 600:]) / 20)
                print("mmWave duration (s):", len(matched_mmwave_data[(i + 1) * 600:]) / 20)
                print()
        elif len(final_ECG) > 220:
            output = os.path.join(outputdir, sub + '_' + str(idx))
            if not os.path.exists(output):
                os.makedirs(output)
            sio.savemat(os.path.join(output, 'ecg.mat'), {'ecg': final_ECG})
            sio.savemat(os.path.join(output, 'resp.mat'), {'resp': final_RESP})
            if len(matched_mmwave_data) > 0:
                sio.savemat(os.path.join(output, 'mmwave.mat'), {'mmwave': matched_mmwave_data})
            print("Physio duration (s):", len(final_ECG) / 20)
            print("mmWave duration (s):", len(matched_mmwave_data) / 20)
            print()

