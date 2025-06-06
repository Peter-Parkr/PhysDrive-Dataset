""" Utilities """

import numpy as np
import argparse
from scipy.fft import fft, fftfreq
from scipy import signal
from scipy.signal import butter, filtfilt, lfilter, welch
import sys
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from einops import rearrange
import torch
import neurokit2 as nk


def Drop_HR(whole_max_idx, delNum=4):
    Row_Num, Individual_Num = whole_max_idx.shape
    HR = []
    for individual in range(Individual_Num):
        HR_sorted = np.sort(whole_max_idx[:, individual])
        HR.append(np.mean(HR_sorted[delNum:-delNum]))
    return np.array(HR)

def slice_bvp(input,slice_lens =11):
    input = rearrange(input, "b c t -> b t c")
    split_slices = []
    for i in range(input.shape[1] + 1 - slice_lens):
        split_slices.append(input[:, i:i + slice_lens, :].unsqueeze(1))
    split_result = torch.cat(split_slices, 1)
    split_result = rearrange(split_result, "b s c t -> b s (c t)")
    return split_result


def cal_cos_similarity_self(tensor):
    # 归一化特征向量以便计算余弦相似度
    norm_tensor = torch.nn.functional.normalize(tensor, p=2, dim=2)  # L2 归一化

    # 计算余弦相似度
    similarity_matrix = torch.matmul(norm_tensor, norm_tensor.transpose(1, 2))

    return similarity_matrix


def hr_fft(sig, fs=20, harmonics_removal=True):
    # get heart rate by FFT
    # return both heart rate and PSD

    sig = sig.reshape(-1)
    sig = sig * signal.windows.hann(sig.shape[0])
    sig_f = np.abs(fft(sig))
    low_idx = np.round(0.6 / fs * sig.shape[0]).astype('int')
    high_idx = np.round(3 / fs * sig.shape[0]).astype('int')
    sig_f_original = sig_f.copy()

    sig_f[:low_idx] = 0
    sig_f[high_idx:] = 0

    peak_idx, _ = signal.find_peaks(sig_f)
    sort_idx = np.argsort(sig_f[peak_idx])
    sort_idx = sort_idx[::-1]

    if len(sort_idx) == 0:
        return 0, sig_f_original, None
    elif len(sort_idx) < 2:
        peak_idx1 = peak_idx[sort_idx[0]]
        f_hr1 = peak_idx1 / sig.shape[0] * fs
        hr1 = f_hr1 * 60
        x_hr = np.arange(len(sig)) / len(sig) * fs * 60
        return hr1, sig_f_original, x_hr

    peak_idx1 = peak_idx[sort_idx[0]]
    peak_idx2 = peak_idx[sort_idx[1]]

    f_hr1 = peak_idx1 / sig.shape[0] * fs
    hr1 = f_hr1 * 60

    f_hr2 = peak_idx2 / sig.shape[0] * fs
    hr2 = f_hr2 * 60
    if harmonics_removal:
        if np.abs(hr1 - 2 * hr2) < 10:
            hr = hr2
        else:
            hr = hr1
    else:
        hr = hr1

    x_hr = np.arange(len(sig)) / len(sig) * fs * 60
    return hr, sig_f_original, x_hr


def rr_fft(sig, fs=20, harmonics_removal=True):
    """
    返回:
      rr: 呼吸频率（breaths per minute）
      sig_f_original: FFT后的原始幅值谱
      x_resp: 对应的频率刻度（单位：breaths per minute）
    """
    sig = sig.reshape(-1)
    sig = sig * signal.windows.hann(sig.shape[0])
    sig_f = np.abs(fft(sig))
    N = sig.shape[0]

    # 对应呼吸频率范围设定：一般成人呼吸频率在 0.1~0.7 Hz 之间，
    # 转换为 bpm 后大致为 6 ~ 42 bpm，这里我们采用频率带 [0.1, 0.7] Hz
    low_idx = np.round(0.1 / fs * N).astype('int')
    high_idx = np.round(0.7 / fs * N).astype('int')
    sig_f_original = sig_f.copy()

    sig_f[:low_idx] = 0
    sig_f[high_idx:] = 0

    peak_idx, _ = signal.find_peaks(sig_f)
    sort_idx = np.argsort(sig_f[peak_idx])
    sort_idx = sort_idx[::-1]

    if len(sort_idx) == 0:
        return 10, sig_f_original, None
    elif len(sort_idx) < 2:
        peak_idx1 = peak_idx[sort_idx[0]]
        f_resp1 = peak_idx1 / N * fs
        rr1 = f_resp1 * 60
        x_resp = np.arange(N) / N * fs * 60
        return rr1, sig_f_original, x_resp

    peak_idx1 = peak_idx[sort_idx[0]]
    peak_idx2 = peak_idx[sort_idx[1]]

    f_resp1 = peak_idx1 / N * fs
    rr1 = f_resp1 * 60

    f_resp2 = peak_idx2 / N * fs
    rr2 = f_resp2 * 60

    if harmonics_removal:
        # 若 rr1 为 rr2 的2倍（或接近），则认为 rr2 更合理
        if np.abs(rr1 - 2 * rr2) < 5:
            rr = rr2
        else:
            rr = rr1
    else:
        rr = rr1

    x_resp = np.arange(N) / N * fs * 60
    return rr, sig_f_original, x_resp

def time_to_str(t, mode='min'):
    if mode == 'min':
        t = int(t) / 60
        hr = t // 60
        min = t % 60
        return '%2d hr %02d min' % (hr, min)
    elif mode == 'sec':
        t = int(t)
        min = t // 60
        sec = t % 60
        return '%2d min %02d sec' % (min, sec)
    else:
        raise NotImplementedError


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None:
            mode = 'w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1):
        if '\r' in message:
            is_file = 0
        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def get_args():
    parser = argparse.ArgumentParser(description='Train ',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 训练参数
    parser.add_argument('-g', '--GPU', dest='GPU', type=str, default='0',
                        help='the index of GPU')
    parser.add_argument('-p', '--pp', dest='num_workers', type=int, default=4,
                        help='num_workers')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=64 * 64 * 64,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=100,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-m', '--model', dest='m', type=str, default='mmFormer',
                        help='model type')
    parser.add_argument('-fn', '--fold_num', type=int, default=5,
                        help='fold_num', dest='fold_num')
    parser.add_argument('-fi', '--fold_index', type=int, default=0,
                        help='fold_index:0-fold_num', dest='fold_index')
    parser.add_argument('-rT', '--reTrain', dest='reTrain', type=int, default=0,
                        help='Load model')
    parser.add_argument('-rD', '--reData', dest='reData', type=int, default=0,
                        help='re Data')
    parser.add_argument('-mi', '--max_iter', dest='max_iter', type=int, default=20000,
                        help='re Data')
    parser.add_argument('-s', '--seed', dest='seed', type=int, default=0,
                        help='seed')

    parser.add_argument('-testper', '--test_percent', type=int, default=20,
                        help='test_percent', dest='test_percent')

    parser.add_argument('-tr', '--temporal_aug_rate', type=float, default=0.5,
                        help='temporal_aug_rate', dest='temporal_aug_rate')
    parser.add_argument('-sr', '--spatial_aug_rate', type=float, default=0.0,
                        help='spatial_aug_rate', dest='spatial_aug_rate')

    # 图片参数
    parser.add_argument('-f', '--form', dest='form', type=str, default='Resize',
                        help='the form of input img')
    parser.add_argument('-dr', '--num_doppler', dest='num_doppler', type=int, default=32,
                        help='the number of doppler')
    parser.add_argument('-ag', '--num_angles', dest='num_angles', type=int, default=32,
                        help='the number of angles')
    parser.add_argument('-rg', '--num_ranges', dest='num_ranges', type=int, default=32,
                        help='the number of ranges')
    parser.add_argument('-n', '--frames_num', dest='frames_num', type=int, default=200,
                        help='the num of frames')
    parser.add_argument('-t', '--tgt', dest='tgt', type=str, default='Physdrive',
                        help='the name of target dataset: Physdrive...')
    return parser.parse_args()


def MyEval(HR_pr, HR_rel):
    HR_pr = np.array(HR_pr).reshape(-1)
    HR_rel = np.array(HR_rel).reshape(-1)
    # HR_pr = (HR_pr - np.min(HR_pr)) / (np.max(HR_pr) - np.min(HR_pr))
    # HR_rel = (HR_rel - np.min(HR_rel)) / (np.max(HR_rel) - np.min(HR_rel))
    temp = HR_pr - HR_rel
    me = np.mean(temp)
    std = np.std(temp)
    mae = np.sum(np.abs(temp)) / len(temp)
    rmse = np.sqrt(np.sum(np.power(temp, 2)) / len(temp))
    mer = np.mean(np.abs(temp) / HR_rel)
    p = np.sum((HR_pr - np.mean(HR_pr)) * (HR_rel - np.mean(HR_rel))) / (
            0.01 + np.linalg.norm(HR_pr - np.mean(HR_pr), ord=2) * np.linalg.norm(HR_rel - np.mean(HR_rel), ord=2))
    # print('| me: %.4f' % me,
    #       '| std: %.4f' % std,
    #       '| mae: %.4f' % mae,
    #       '| rmse: %.4f' % rmse,
    #       '| mer: %.4f' % mer,
    #       '| p: %.4f' % p
    #       )
    return me, std, mae, rmse, mer, p


def preprocess_signal(bvp_signal, fps):
    # High-pass filter to remove baseline wander
    b, a = signal.butter(1, 0.5 / (fps / 2), btype='high')
    filtered_signal = signal.filtfilt(b, a, bvp_signal)
    return filtered_signal





def rr_cal(resp_signal, fps=30):
    # Preprocess the signal
    b, a = signal.butter(1, 0.5 / (fps / 2), btype='high')
    preprocessed_signal = signal.filtfilt(b, a, resp_signal)

    # Find peaks
    peaks, _ = signal.find_peaks(preprocessed_signal,
                                 distance=(fps / 20) * 60)  # Assuming a minimum heart rate of 40 bpm
    if len(peaks) < 2:  # Need at least two peaks to calculate heart rate
        return np.float64(12.0)

    # Calculate peak intervals and heart rate
    peak_intervals = np.diff(peaks) / fps
    rr = 60 / peak_intervals.mean()

    return rr

def hr_cal(signal, fps=30):
    peaks = nk.ecg_findpeaks(signal, sampling_rate=fps, method='rodrigues2021')['ECG_R_Peaks']
    peaks -= 1
    rr_intervals = np.diff(peaks) / fps * 1000  # 将间隔转换为毫秒
    nn_intervals = rr_intervals
    hr = 60000 / np.mean(nn_intervals)
    return hr


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Applies a Butterworth bandpass filter to the data.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y


def fft_spectrum(data, fs):
    """
    Calculates the FFT spectrum and frequencies.
    """
    N = len(data)
    yf = np.fft.fft(data)
    xf = np.linspace(0.0, 1.0 / (2.0 / fs), N // 2)
    psd = 2.0 / N * np.abs(yf[0:N // 2])
    return xf, psd


def MyEval_ecg_hr(ecg_pr, ecg_rel):
    HR_pr, HR_rel = [], []
    for i in range(len(ecg_pr)):
        ecg = np.array(ecg_pr[i]).reshape(-1)
        ecg = (ecg - np.min(ecg)) / (np.max(ecg) - np.min(ecg))
        ecg = ecg.astype('float32')
        res = hr_cal(ecg)
        HR_pr.append(res)
        ecg = np.array(ecg_rel[i]).reshape(-1)
        ecg = (ecg - np.min(ecg)) / (np.max(ecg) - np.min(ecg))
        ecg = ecg.astype('float32')
        res = hr_cal(ecg)
        HR_rel.append(res)
    HR_pr = np.array(HR_pr)  # .reshape(-1)
    HR_pr[HR_pr == 0] = np.mean(HR_pr[HR_pr != 0])
    HR_rel = np.array(HR_rel)  # .reshape(-1)
    HR_rel[HR_rel == 0] = np.mean(HR_rel[HR_rel != 0])
    temp = HR_pr - HR_rel
    me = np.mean(temp)
    std = np.std(temp)
    mae = np.sum(np.abs(temp)) / len(temp)
    rmse = np.sqrt(np.sum(np.power(temp, 2)) / len(temp))
    mer = np.mean(np.abs(temp) / HR_rel)
    p = np.sum((HR_pr - np.mean(HR_pr)) * (HR_rel - np.mean(HR_rel))) / (
            0.01 + np.linalg.norm(HR_pr - np.mean(HR_pr), ord=2) * np.linalg.norm(HR_rel - np.mean(HR_rel), ord=2))
    return me, std, mae, rmse, mer, p


def MyEval_resp_rr(resp_pr, resp_rel):
    RR_pr, RR_rel = [], []
    for i in range(len(resp_pr)):
        resp = np.array(resp_pr[i]).reshape(-1)
        resp = (resp - np.min(resp)) / (np.max(resp) - np.min(resp))
        resp = resp.astype('float32')
        res = rr_cal(resp)
        RR_pr.append(res)
        resp = np.array(resp_rel[i]).reshape(-1)
        resp = (resp - np.min(resp)) / (np.max(resp) - np.min(resp))
        resp = resp.astype('float32')
        res = rr_cal(resp)
        RR_rel.append(res)
    RR_pr = np.array(RR_pr)  # .reshape(-1)
    RR_pr[RR_pr == 0] = np.mean(RR_pr[RR_pr != 0])
    RR_rel = np.array(RR_rel)  # .reshape(-1)
    RR_rel[RR_rel == 0] = np.mean(RR_rel[RR_rel != 0])
    temp = RR_pr - RR_rel
    me = np.mean(temp)
    std = np.std(temp)
    mae = np.sum(np.abs(temp)) / len(temp)
    rmse = np.sqrt(np.sum(np.power(temp, 2)) / len(temp))
    mer = np.mean(np.abs(temp) / RR_rel)
    p = np.sum((RR_pr - np.mean(RR_pr)) * (RR_rel - np.mean(RR_rel))) / (
            0.01 + np.linalg.norm(RR_pr - np.mean(RR_pr), ord=2) * np.linalg.norm(RR_rel - np.mean(RR_rel), ord=2))
    return me, std, mae, rmse, mer, p



def param_size(model):
    """ Compute parameter size in MB """
    n_params = sum(
        np.prod(v.size()) for k, v in model.named_parameters() if not k.startswith('aux_head'))
    return n_params / 1024. / 1024.


def loss_visual(loss_res, save_path, multi_dataset=None):
    if multi_dataset is not None:
        for key in loss_res.keys():
            # draw the loss curve, x-axis is the number of iterations, y-axis is the loss value
            save_path_temp = save_path.replace('key', key)
            plt.figure()
            tag_emp = True
            for d in multi_dataset.keys():
                if len(multi_dataset[d][key]) > 0:
                    tag_emp = False
                plt.plot(multi_dataset[d][key], label=d)
            if tag_emp:
                continue
            plt.title('task loss for ' + key, fontdict={'family': 'Times New Roman', 'size': 15})
            plt.xlabel('iteration', fontdict={'family': 'Times New Roman', 'size': 10})
            plt.ylabel('loss', fontdict={'family': 'Times New Roman', 'size': 10})
            plt.legend(prop={'size': 10, 'family': 'Times New Roman', 'weight': 'bold'})
            plt.savefig(save_path_temp)
    else:
        for key in loss_res.keys():
            # draw the loss curve, x-axis is the number of iterations, y-axis is the loss value
            if len(loss_res[key]) < 1:
                continue
            save_path_temp = save_path.replace('key', key)
            plt.figure()
            plt.plot(loss_res[key])
            plt.title('task loss for ' + key, fontdict={'family': 'Times New Roman', 'size': 15})
            plt.xlabel('iteration', fontdict={'family': 'Times New Roman', 'size': 10})
            plt.ylabel('loss', fontdict={'family': 'Times New Roman', 'size': 10})
            plt.savefig(save_path_temp)
