""" Utilities """
import os
import shutil
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function
import numpy as np
import argparse
from torch.autograd import Variable
from numpy import random
import math
import utils
from scipy.signal import find_peaks, welch
from scipy import signal
from scipy.fft import fft
# from SNiC import ideal_bandpass, normalize_psd, IPR_SSL, EMD_SSL, torch_power_spectral_density, SNR_SSL

args = utils.get_args()


class P_loss3(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pre_lable, gt_lable):
        if len(gt_lable.shape) == 3:
            M, N, A = gt_lable.shape
            gt_lable = gt_lable - torch.mean(gt_lable, dim=2).view(M, N, 1)
            pre_lable = pre_lable - torch.mean(pre_lable, dim=2).view(M, N, 1)
        aPow = torch.sqrt(torch.sum(torch.mul(gt_lable, gt_lable), dim=-1))
        bPow = torch.sqrt(torch.sum(torch.mul(pre_lable, pre_lable), dim=-1))
        pearson = torch.sum(torch.mul(gt_lable, pre_lable), dim=-1) / (aPow * bPow + 0.001)
        loss = 1 - torch.sum(pearson, dim=0) / (gt_lable.shape[0])
        '''
        _, psd = torch_power_spectral_density(gt_lable, fps=30, low_hz=40 / 60, high_hz=180 / 60,
                                              normalize=True, bandpass=True, device=gt_lable.device)
        _, psd_pre = torch_power_spectral_density(pre_lable, fps=30, low_hz=40 / 60, high_hz=180 / 60,
                                                  normalize=True, bandpass=True, device=gt_lable.device)
        l_psd = F.kl_div(psd_pre.log(), psd, reduction='mean')
        loss += l_psd'''
        return loss
# class P_loss3(nn.Module):
#     """
#     The Neg_Pearson Module is from the orignal author of Physnet.
#     Code of 'Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks'
#     source: https://github.com/ZitongYu/PhysNet/blob/master/NegPearsonLoss.py
#     """
#
#     def __init__(self):
#         super(P_loss3, self).__init__()
#         return
#
#     def forward(self, preds, labels):
#         loss = 0
#         for i in range(preds.shape[0]):
#             sum_x = torch.sum(preds[i])
#             sum_y = torch.sum(labels[i])
#             sum_xy = torch.sum(preds[i] * labels[i])
#             sum_x2 = torch.sum(torch.pow(preds[i], 2))
#             sum_y2 = torch.sum(torch.pow(labels[i], 2))
#             N = preds.shape[1]
#             pearson = (N * sum_xy - sum_x * sum_y) / (
#                 torch.sqrt((N * sum_x2 - torch.pow(sum_x, 2)) * (N * sum_y2 - torch.pow(sum_y, 2))))
#             loss += 1 - pearson
#
#         loss = loss / preds.shape[0]
#         return loss

class ContrastLoss(nn.Module):

    def __init__(self):
        super(ContrastLoss, self).__init__()
        pass

    def forward(self, anchor_fea, reassembly_fea, contrast_label):
        anchor_fea = anchor_fea.unsqueeze(0).repeat(reassembly_fea.shape[0], 1)
        sim_matrix = F.cosine_similarity(anchor_fea, reassembly_fea)
        loss = -1 * sim_matrix * contrast_label
        return loss.mean()


class CovarianceConsistentLoss(nn.Module):

    def __init__(self):
        super(CovarianceConsistentLoss, self).__init__()
        pass

    def forward(self, avs, avs_aug, labels=None):
        # label_mx = self.compute_custom_distance_matrix(labels)
        loss = 0
        for i in range(len(avs)):
            cov_matrix = torch.matmul(avs[i], avs[i].transpose(0, 1))
            cov_matrix_aug = torch.matmul(avs_aug[i], avs_aug[i].transpose(0, 1))
            loss += torch.norm(cov_matrix - cov_matrix_aug, p='fro')
        return loss
        # torch.bmm(avs.permute(0, 2, 1)
        # sim_matrix = F.cosine_similarity(avs, avs_aug)
        # loss = -1 * sim_matrix * contrast_label
        # return loss.mean()


class CrossCovarianceLoss(nn.Module):

    def __init__(self):
        super(CrossCovarianceLoss, self).__init__()
        pass

    def forward(self, avs, avs_aug, labels=None):
        # label_mx = self.compute_custom_distance_matrix(labels)
        loss = 0
        for i in range(len(avs)):
            cov_matrix = torch.matmul(avs[i], avs_aug[i].transpose(0, 1))
            # 将cov_matrix的对角线元素相加
            loss += torch.sum(torch.diag(cov_matrix) - 1)
        return loss


def RobustLoss(consist_loss_signal, dissimilarity_loss, optimizer, params):
    optimizer.zero_grad(set_to_none=True)
    consist_loss_signal.backward(retain_graph=True)
    grads1 = [p.grad for p in params]

    for p in params:
        p.grad = None

    # 计算第二个损失的梯度
    optimizer.zero_grad(set_to_none=True)
    dissimilarity_loss.backward(retain_graph=True)
    grads2 = [p.grad for p in params]

    grad_norm1 = sum([g.norm() for g in grads1 if g is not None])
    grad_norm2 = sum([g.norm() for g in grads2 if g is not None])

    # 动态调整权重
    weight1 = grad_norm2 / (grad_norm1 + grad_norm2)
    weight2 = grad_norm1 / (grad_norm1 + grad_norm2)

    dot_product = sum((g1 * g2).sum() for g1, g2 in zip(grads1, grads2) if g1 is not None and g2 is not None)
    # print(dot_product)
    for p in params:
        p.grad = None
    # 判断夹角
    if dot_product < 0:  # 夹角大于90度
        # print('xiaoyu')
        return consist_loss_signal + dissimilarity_loss
    else:
        return weight1 * consist_loss_signal + weight2 * dissimilarity_loss


class ConsistencyLoss(nn.Module):

    def __init__(self):
        super(ConsistencyLoss, self).__init__()
        self.CL_hr = consistent_loss(thr=0)
        self.CL_other = consistent_loss(thr=0)

    def forward(self, hr, hr_aug, rr, rr_aug, spo, spo_aug, bvp, bvp_aug, loss_hr, loss_rr, loss_spo, loss_sig,
                inter_num):
        # label_mx = self.compute_custom_distance_matrix(labels)
        # self.CL = self.CL.to(hr.device)
        # k = 2.0 / (1.0 + np.exp(-10.0 * inter_num / args.max_iter)) - 1.0

        # l_sig = loss_sig(bvp, bvp_aug)
        # l_psd = loss_sig(bvp, bvp_aug)

        freqs, psd = torch_power_spectral_density(bvp, fps=30, low_hz=40 / 60, high_hz=180 / 60,
                                                  normalize=True, bandpass=True, device=bvp.device)
        freqs_aug, psd_aug = torch_power_spectral_density(bvp_aug, fps=30, low_hz=40 / 60, high_hz=180 / 60,
                                                          normalize=True, bandpass=True, device=bvp_aug.device)
        l_psd = F.kl_div(psd_aug.log(), psd,
                         reduction='mean')  # self.sinkhorn_distance(psd.squeeze(), psd_aug.squeeze())
        l_hr = self.CL_hr(hr, hr_aug)
        if args.tgt in ['VIPL', 'PURE', 'VV100']:
            l_spo = self.CL_other(spo, spo_aug)
            loss_metrics = l_hr + l_spo
        elif args.tgt in ['V4V', 'HCW']:
            l_rr = self.CL_other(rr, rr_aug)
            loss_metrics = l_hr + l_rr
        elif args.tgt in [ 'HMPC-Dv1']:
            l_rr = self.CL_other(rr, rr_aug)
            l_spo = self.CL_other(spo, spo_aug)
            loss_metrics = l_hr + l_rr + l_spo
        else:
            loss_metrics = l_hr
        return loss_metrics, l_psd


def orthogonal_loss(A, B):
    AB_T = torch.matmul(A.t(), B)

    I = torch.eye(AB_T.size(0), device=AB_T.device)

    loss = torch.norm(AB_T - I, p='fro')

    return loss


def new_style_alignment_loss(Fx, Fxa, alpha=0.5):
    reconstruction_loss = 0
    correlation_enhancement = 0
    # Ux, Sx, Vx = torch.linalg.svd(torch.cat(Fx, dim=1))
    # Uxa, Sxa, Vxa = torch.linalg.svd(torch.cat(Fxa, dim=1))
    # reconstruction_loss = torch.norm(Sx - Sxa, p='fro')
    for i in range(len(Fx)):
        # 计算 SVD 分解
        cov_matrix = torch.matmul(Fx[i], Fx[i].transpose(0, 1))
        cov_matrix_aug = torch.matmul(Fxa[i], Fxa[i].transpose(0, 1))
        correlation_enhancement += torch.norm(cov_matrix - cov_matrix_aug, p='fro')
        Ux, Sx, Vx = torch.svd(Fx[i])
        Uxa, Sxa, Vxa = torch.svd(Fxa[i])

        Sx_diag = torch.diag_embed(Sx)  # 转换为对角矩阵
        Sxa_diag = torch.diag_embed(Sxa)
        reconstruction_loss += torch.norm(Ux @ Sxa_diag @ Vx.transpose(0, 1) - Uxa @ Sx_diag @ Vxa.transpose(0, 1),
                                          p='fro')
        # reconstruction_loss = F.mse_loss(Ux @ Vxa.t(), Uxa @ Vx.t())
        # cov_matrix = torch.matmul(Fx[i], Fxa[i].transpose(0, 1))
        # # 将cov_matrix的对角线元素相加
        # reconstruction_loss += torch.sum(torch.diag(cov_matrix) - 1)

        # 相关性增强计算
        # reconstruction_loss += torch.norm(Sx - Sxa, p='fro')

    # 总损失

    return reconstruction_loss, correlation_enhancement


def cosine_similarity_loss(matrix1, matrix2):
    # 重新调整矩阵形状为 (batch, -1) 以计算余弦相似度
    matrix1_flat = matrix1.view(matrix1.size(0), -1)
    matrix2_flat = matrix2.view(matrix2.size(0), -1)

    # 计算余弦相似度
    cosine_sim = F.cosine_similarity(matrix1_flat, matrix2_flat, dim=1)

    # 余弦相似度越接近1，表示越相似，损失为1 - cosine_sim
    loss = cosine_sim.mean()

    return loss


class SP_loss(nn.Module):
    def __init__(self, device, clip_length=256, delta=3, loss_type=1, use_wave=False):
        super(SP_loss, self).__init__()

        self.clip_length = clip_length
        self.time_length = clip_length
        self.device = device
        self.delta = delta
        self.delta_distribution = [0.4, 0.25, 0.05]
        self.low_bound = 40
        self.high_bound = 150

        self.bpm_range = torch.arange(self.low_bound, self.high_bound, dtype=torch.float).to(self.device)
        self.bpm_range = self.bpm_range / 60.0

        self.pi = 3.14159265
        two_pi_n = Variable(2 * self.pi * torch.arange(0, self.time_length, dtype=torch.float))
        hanning = Variable(torch.from_numpy(np.hanning(self.time_length)).type(torch.FloatTensor),
                           requires_grad=True).view(1, -1)

        self.two_pi_n = two_pi_n.to(self.device)
        self.hanning = hanning.to(self.device)

        self.cross_entropy = nn.CrossEntropyLoss()
        self.nll = nn.NLLLoss()
        self.l1 = nn.L1Loss()

        self.loss_type = loss_type
        self.eps = 0.0001

        self.lambda_l1 = 0.1
        self.use_wave = use_wave

    def forward(self, wave, gt, pred=None, flag=None):  # all variable operation
        fps = 30

        hr = gt.clone()

        hr[hr.ge(self.high_bound)] = self.high_bound - 1
        hr[hr.le(self.low_bound)] = self.low_bound

        if pred is not None:
            pred = torch.mul(pred, fps)
            pred = pred * 60 / self.clip_length

        batch_size = wave.shape[0]

        f_t = self.bpm_range / fps
        preds = wave * self.hanning

        preds = preds.view(batch_size, 1, -1)
        f_t = f_t.repeat(batch_size, 1).view(batch_size, -1, 1)

        tmp = self.two_pi_n.repeat(batch_size, 1)
        tmp = tmp.view(batch_size, 1, -1)

        complex_absolute = torch.sum(preds * torch.sin(f_t * tmp), dim=-1) ** 2 \
                           + torch.sum(preds * torch.cos(f_t * tmp), dim=-1) ** 2

        target = hr - self.low_bound
        target = target.type(torch.long).view(batch_size)

        whole_max_val, whole_max_idx = complex_absolute.max(1)
        whole_max_idx = whole_max_idx + self.low_bound

        if self.loss_type == 1:
            loss = self.cross_entropy(complex_absolute, target)

        elif self.loss_type == 7:
            norm_t = (torch.ones(batch_size).to(self.device) / torch.sum(complex_absolute, dim=1))
            norm_t = norm_t.view(-1, 1)
            complex_absolute = complex_absolute * norm_t

            loss = self.cross_entropy(complex_absolute, target)

            idx_l = target - self.delta
            idx_l[idx_l.le(0)] = 0
            idx_r = target + self.delta
            idx_r[idx_r.ge(self.high_bound - self.low_bound - 1)] = self.high_bound - self.low_bound - 1;

            loss_snr = 0.0
            for i in range(0, batch_size):
                loss_snr = loss_snr + 1 - torch.sum(complex_absolute[i, idx_l[i]:idx_r[i]])

            loss_snr = loss_snr / batch_size

            loss = loss + loss_snr

        return loss, whole_max_idx


class consistent_loss(nn.Module):
    def __init__(self, thr=5):
        super(consistent_loss, self).__init__()
        self.L1Loss = nn.L1Loss()
        self.thr = thr

    def forward(self, pre, pre_aug):  # all variable operation
        temp = torch.abs(pre - pre_aug)
        return self.L1Loss(torch.where(temp >= self.thr, pre, pre_aug), pre_aug)


def get_HR_from_bvp_torch(signals, fs=30, harmonics_removal=True):
    batch_size, signal_length = signals.shape

    # Apply Hann window
    hann_window = torch.hann_window(signal_length, device=signals.device, requires_grad=True)
    signals = signals * hann_window

    # Compute FFT and magnitude
    sig_f = torch.abs(torch.fft.fft(signals, dim=1))

    # Frequency limits for heart rate detection
    low_idx = int(round(0.6 / fs * signal_length))
    high_idx = int(round(3 / fs * signal_length))

    # Zero out frequencies outside the desired range
    sig_f[:, :low_idx] = 0
    sig_f[:, high_idx:] = 0

    # Find the top two peaks for each signal
    top_two_peaks = torch.topk(sig_f, 2, dim=1).indices

    # Compute heart rates
    peak_idx1 = top_two_peaks[:, 0]
    peak_idx2 = top_two_peaks[:, 1]

    f_hr1 = peak_idx1 / signal_length * fs
    f_hr2 = peak_idx2 / signal_length * fs

    hr1 = f_hr1 * 60
    hr2 = f_hr2 * 60

    # Harmonics removal logic
    if harmonics_removal:
        hr = torch.where(torch.abs(hr1 - 2 * hr2) < 10, hr2, hr1)
    else:
        hr = hr1

    sig_f_original = sig_f.clone()
    x_hr = torch.arange(signal_length) / signal_length * fs * 60

    return hr, sig_f_original, x_hr


class bvp_hr_loss(nn.Module):
    def __init__(self):
        super(bvp_hr_loss, self).__init__()
        self.L1Loss = nn.L1Loss()

    def forward(self, pre_bvp, gt_hr):  # all variable operation
        pre_bvp = pre_bvp.squeeze()
        hrs, _, _ = get_HR_from_bvp_torch(pre_bvp, fs=30, harmonics_removal=True)
        return self.L1Loss(hrs, gt_hr)
        '''
        freqs, psd = torch_power_spectral_density(pre_bvp, fps=30, low_hz=40, high_hz=180,
                                                  normalize=False, bandpass=False, device=pre_bvp.device)
        loss = IPR_SSL(freqs, psd, device=pre_bvp.device, low_hz=40 / 60,
                       high_hz=180 / 60) + EMD_SSL(freqs, psd, device=pre_bvp.device, low_hz=40 / 60,
                                                   high_hz=180 / 60) + SNR_SSL(freqs, psd, device=pre_bvp.device,
                                                                               low_hz=40 / 60,
                                                                               high_hz=180 / 60)
        return loss'''


def get_RR_from_bvp_torch(bvp_signals):
    freqs, psd = torch_power_spectral_density(bvp_signals, fps=30, low_hz=40, high_hz=180,
                                              normalize=False, bandpass=False, device=bvp_signals.device)

    min_freq_index = int(0.1 * 256 / 30)
    max_freq_index = int(0.5 * 256 / 30)

    # 忽略直流分量，找到最大功率对应的频率
    peak_frequency_indices = torch.argmax(psd[:, min_freq_index:max_freq_index], dim=1) + min_freq_index
    peak_frequencies = freqs[peak_frequency_indices] * 60  #
    return peak_frequencies


class bvp_rr_loss(nn.Module):
    def __init__(self, thr=3):
        super(bvp_rr_loss, self).__init__()
        self.L1Loss = nn.L1Loss()
        self.thr = thr

    def forward(self, gt_bvp, pre_rr):  # all variable operation
        gt_bvp = gt_bvp.squeeze(1)
        rrs = get_RR_from_bvp_torch(gt_bvp)
        temp = torch.abs(rrs - pre_rr)
        return self.L1Loss(torch.squeeze(torch.where(temp >= self.thr, pre_rr, rrs)), torch.squeeze(rrs))


class Asp_loss(nn.Module):
    def __init__(self):
        super(Asp_loss, self).__init__()
        self.L1Loss = nn.L1Loss()

    def generate_aug(self, feat_spo, spo, num=5):
        if spo[0] == 0:
            return None, None
        self.batch_size = feat_spo.shape[0]
        device = feat_spo.device
        sample = 1 - 0.1 * torch.rand(dtype=float, size=(self.batch_size, int(num))).to(device)
        spo_aug = spo.unsqueeze(1).repeat(1, int(num))
        spo_aug = torch.mul(spo_aug, sample)
        sample = sample.unsqueeze(-1)
        sample.expand(self.batch_size, int(num), feat_spo.shape[1])
        feat_spo = feat_spo.unsqueeze(1).repeat(1, int(num), 1)
        feat_aug = torch.mul(feat_spo, sample)
        return feat_aug.float(), spo_aug.float()

    def forward(self, spo_pred, spo):  # all variable operation
        spo_pred = spo_pred.squeeze()
        return self.L1Loss(torch.where(spo > 93, spo_pred, spo).view(-1, ), spo.view(-1, ))




def get_loss(bvp_pre, resp_pre, hr_pre, rr_pre, spo_pre, bvp_gt, resp_gt, hr_gt, rr_gt, spo_gt, dataName, loss_bvp, loss_resp, loss_hr,
             loss_rr, loss_spo, args, inter_num, loss_res=None):
    k = 2.0 / (1.0 + np.exp(-10.0 * inter_num / args.max_iter)) - 1.0
    if dataName == 'On_Road_rPPG':
        # l_bvp = loss_bvp[0](bvp_pre, bvp_gt) #+ 0.1 * loss_bvp[1](bvp_pre, bvp_gt)
        #l_resp = loss_resp[0](resp_pre, resp_gt) #+ 0.1 * loss_resp[1](resp_pre, resp_gt)
        l_bvp = loss_bvp[0](bvp_pre, bvp_gt) #+ 0.1 * loss_bvp[1](bvp_pre, bvp_gt)
        l_hr =  loss_hr(torch.squeeze(hr_pre), hr_gt) / 10
        l_rr = loss_rr(torch.squeeze(rr_pre), rr_gt) / 10
        l_spo = loss_spo(torch.squeeze(spo_pre), spo_gt) / 10
        loss = (l_bvp + l_rr + l_hr) / 3
        # loss = l_hr
        if loss_res is not None:
            # loss_res['bvp'].append(l_bvp.item())
            #loss_res['resp'].append(l_resp.item())
            loss_res['hr'].append(l_hr.item())
            loss_res['rf'].append(l_rr.item())
            loss_res['spo'].append(l_spo.item())
            loss_res['all'].append(loss.item())
    if dataName == 'BUAA':
        l_bvp = loss_bvp[0](bvp_pre, bvp_gt) + 0.1 * loss_bvp[1](bvp_pre, bvp_gt)
        #l_resp = loss_resp[0](resp_pre, resp_gt) #+ 0.1 * loss_resp[1](resp_pre, resp_gt)
        l_hr = k * loss_hr(torch.squeeze(hr_pre), hr_gt) / 10
        # l_rr = k * loss_rr(torch.squeeze(rr_pre), rr_gt) / 10
        # l_spo = k * loss_spo(torch.squeeze(spo_pre), spo_gt) / 10
        loss = (l_bvp + l_hr) / 2
        if loss_res is not None:
            loss_res['bvp'].append(l_bvp.item())
            #loss_res['resp'].append(l_resp.item())
            loss_res['hr'].append(l_hr.item())
            # loss_res['rf'].append(l_rr.item())
            # loss_res['spo'].append(l_spo.item())
            loss_res['all'].append(loss.item())

    return loss, loss_res
