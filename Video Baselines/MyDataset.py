# -*- coding: UTF-8 -*-
import numpy as np
import os

import pandas as pd
from torch.utils.data import Dataset
import cv2
import csv
import scipy.io as scio
from scipy.signal import find_peaks, butter, filtfilt
import torchvision.transforms.functional as transF
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from utils import rr_cal
import random
import utils
import re
from scipy.signal import butter
from scipy.sparse import spdiags
from typing import List, Dict



normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

class Data_Video(Dataset):
    def __init__(self, root_dir, dataName, STMap, frames_num, args, transform=None, domain_label=None, datalist=None):
        self.root_dir = root_dir
        self.dataName = dataName
        self.STMap_Name = STMap
        self.frames_num = int(frames_num)
        if datalist is not None:
            self.datalist = datalist
        else:
            self.datalist = os.listdir(root_dir)
        self.datalist = sorted(self.datalist)
        self.num = len(self.datalist)
        self.domain_label = domain_label
        self.transform = transform
        self.args = args
        self.temp = None


    def __len__(self):
        return self.num

    def getLabel(self, nowPath, Step_Index):
        # 读取每个数据集的心率值和BVP信号
        if self.dataName == 'VV100':
            bvp_name = 'Label/BVP_Filt.mat'
            bvp_path = os.path.join(nowPath, bvp_name)
            bvp = scio.loadmat(bvp_path)['BVP']
            bvp = np.array(bvp.astype('float32')).reshape(-1)
            bvp = bvp[Step_Index:Step_Index + self.frames_num]
            bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))
            bvp = bvp.astype('float32')

            gt_name = 'Label/HR.mat'
            gt_path = os.path.join(nowPath, gt_name)
            gt = scio.loadmat(gt_path)['HR']
            gt = np.array(gt.astype('float32')).reshape(-1)
            gt = gt[int(Step_Index / 10)]
            gt = gt.astype('float32').reshape(-1)

            sp_name = 'Label/SPO2.mat'
            sp_path = os.path.join(nowPath, sp_name)
            sp = scio.loadmat(sp_path)['HR']  # wrong label name but correct value
            sp = np.array(sp.astype('float32')).reshape(-1)
            sp = np.nanmean(sp[Step_Index:Step_Index + self.frames_num])
            sp = sp.astype('float32')
            return gt, bvp, sp

        elif self.dataName == 'MMPD':
            bvp_name = 'Label/BVP_Filt.mat'
            bvp_path = os.path.join(nowPath, bvp_name)
            bvp = scio.loadmat(bvp_path)['BVP']
            bvp = np.array(bvp.astype('float32')).reshape(-1)
            bvp = bvp[Step_Index:Step_Index + self.frames_num]
            bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))
            bvp = bvp.astype('float32')

            gt, _, _ = utils.hr_fft(bvp, fs=30, harmonics_removal=True)
            gt = np.array(gt).reshape(-1)
            gt = gt.astype('float32')


        elif self.dataName == 'BUAA':
            bvp_name = 'Label/BVP.mat'
            bvp_path = os.path.join(nowPath, bvp_name)
            bvp = scio.loadmat(bvp_path)['BVP']
            bvp = np.array(bvp.astype('float32')).reshape(-1)
            bvp = bvp[Step_Index:Step_Index + self.frames_num]
            bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))
            bvp = bvp.astype('float32')

            gt_name = 'Label/HR_256.mat'
            gt_path = os.path.join(nowPath, gt_name)
            gt = scio.loadmat(gt_path)['HR']
            gt = np.array(gt.astype('float32')).reshape(-1)
            gt = gt[int(Step_Index / 10)]
            gt = gt.astype('float32').reshape(-1)

        elif self.dataName == 'VIPL':
            bvp_name = 'Label_CSI/BVP_Filt.mat'
            bvp_path = os.path.join(nowPath, bvp_name)
            bvp = scio.loadmat(bvp_path)['BVP']
            bvp = np.array(bvp.astype('float32')).reshape(-1)
            bvp = bvp[Step_Index:Step_Index + self.frames_num]
            bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))
            bvp = bvp.astype('float32')

            gt_name = 'Label_CSI/HR.mat'
            gt_path = os.path.join(nowPath, gt_name)
            gt = scio.loadmat(gt_path)['HR']
            gt = np.array(gt.astype('float32')).reshape(-1)
            gt = np.nanmean(gt[Step_Index:Step_Index + self.frames_num])
            gt = gt.astype('float32').reshape(-1)

            sp_name = 'Label_CSI/SPO2.mat'
            sp_path = os.path.join(nowPath, sp_name)
            sp = scio.loadmat(sp_path)['SPO2']
            sp = np.array(sp.astype('float32')).reshape(-1)
            sp = np.nanmean(sp[Step_Index:Step_Index + self.frames_num])
            sp = sp.astype('float32')

            return gt, bvp, sp

        elif self.dataName == 'V4V':
            gt_name = 'Label/HR.mat'
            gt_path = os.path.join(nowPath, gt_name)
            gt = scio.loadmat(gt_path)['HR']
            gt = np.array(gt.astype('float32')).reshape(-1)
            gt = np.nanmean(gt[Step_Index:Step_Index + self.frames_num])
            gt = gt.astype('float32').reshape(-1)
            bvp = np.array(0.0)
            bvp = bvp.astype('float32')

            rf_name = 'Label/RF.mat'
            rf_path = os.path.join(nowPath, rf_name)
            rf = scio.loadmat(rf_path)['RF']
            rf = np.array(rf.astype('float32')).reshape(-1)
            rf = np.nanmean(rf[Step_Index:Step_Index + self.frames_num])
            rf = rf.astype('float32')

            resp = np.array(0.0)
            resp = resp.astype('float32')

            return gt, bvp, rf, resp

        elif self.dataName == 'HCW':
            bvp_name = 'Label/BVP_Filt.mat'
            bvp_path = os.path.join(nowPath, bvp_name)
            bvp = scio.loadmat(bvp_path)['BVP']
            bvp = np.array(bvp.astype('float32')).reshape(-1)
            bvp = bvp[Step_Index:Step_Index + self.frames_num]
            bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))
            bvp = bvp.astype('float32')

            gt_name = 'Label/HR_Filt.mat'
            gt_path = os.path.join(nowPath, gt_name)
            gt = scio.loadmat(gt_path)['HR']
            gt = np.array(gt.astype('float32')).reshape(-1)
            gt = gt[Step_Index]  # np.nanmean(gt[Step_Index:Step_Index + self.frames_num])
            gt = gt.astype('float32')
            '''
            gt, _, _ = utils.hr_fft(bvp, fs=30, harmonics_removal=True)
            gt = np.array(gt).reshape(-1)
            gt = gt.astype('float32')'''

            rf_name = 'Label/RF_Filt.mat'
            rf_path = os.path.join(nowPath, rf_name)
            rf = scio.loadmat(rf_path)['RF']
            rf = np.array(rf.astype('float32')).reshape(-1)
            rf = rf[Step_Index]  # np.nanmean(rf[Step_Index:Step_Index + self.frames_num])
            rf = rf.astype('float32')

            resp_name = 'Label/RESP_Filt.mat'
            resp_path = os.path.join(nowPath, resp_name)
            resp = scio.loadmat(resp_path)['RESP']
            resp = np.array(resp.astype('float32')).reshape(-1)
            resp = resp[Step_Index:Step_Index + self.frames_num]
            resp = (resp - np.min(resp)) / (np.max(resp) - np.min(resp))
            resp = resp.astype('float32')

            return gt, bvp, rf, resp

        elif self.dataName == 'PURE':
            bvp_name = 'Label/BVP_Filt.mat'
            bvp_path = os.path.join(nowPath, bvp_name)
            bvp = scio.loadmat(bvp_path)['BVP']
            bvp = np.array(bvp.astype('float32')).reshape(-1)
            bvp = bvp[Step_Index:Step_Index + self.frames_num]
            bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))
            bvp = bvp.astype('float32')

            gt = utils.hr_cal(bvp)
            gt = np.array(gt)
            gt = gt.astype('float32')

            sp_name = 'Label/SPO2.mat'
            sp_path = os.path.join(nowPath, sp_name)
            sp = scio.loadmat(sp_path)['SPO2']
            sp = np.array(sp.astype('float32')).reshape(-1)
            sp = np.nanmean(sp[Step_Index:Step_Index + self.frames_num])
            sp = sp.astype('float32')
            return gt, bvp, sp

        elif self.dataName == 'On-Road-rPPG':
            bvp_name = 'Label/BVP_Filt.mat'
            bvp_path = os.path.join(nowPath, bvp_name)
            bvp = scio.loadmat(bvp_path)['BVP']
            # bvp_name = 'Label/ECG.mat'
            # bvp_path = os.path.join(nowPath, bvp_name)
            # bvp = scio.loadmat(bvp_path)['ECG']
            # def _detrend(input_signal, lambda_value=100):
            #     """Detrend PPG signal."""
            #     signal_length = input_signal.shape[0]
            #     # observation matrix
            #     H = np.identity(signal_length)
            #     ones = np.ones(signal_length)
            #     minus_twos = -2 * np.ones(signal_length)
            #     diags_data = np.array([ones, minus_twos, ones])
            #     diags_index = np.array([0, 1, 2])
            #     D = spdiags(diags_data, diags_index,
            #                 (signal_length - 2), signal_length).toarray()
            #     detrended_signal = np.dot(
            #         (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
            #     return detrended_signal
            bvp = np.array(bvp.astype('float32')).reshape(-1)
            bvp = bvp[Step_Index:Step_Index + self.frames_num]
            bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))
            bvp = bvp.astype('float32')
            # bvp = _detrend(bvp)
            # bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))

            # gt_name = 'Label/HR.mat'
            # gt_path = os.path.join(nowPath, gt_name)
            # gt = scio.loadmat(gt_path)['HR']
            # gt = np.array(gt.astype('float32')).reshape(-1)
            # gt = np.nanmean(gt[Step_Index:Step_Index + self.frames_num])
            # gt = gt.astype('float32').reshape(-1)
            gt = utils.hr_cal(bvp)
            gt = np.array(gt)
            gt = gt.astype('float32')
            # print(gt)

            sp_name = 'Label/SPO2.mat'
            sp_path = os.path.join(nowPath, sp_name)
            sp = scio.loadmat(sp_path)['SPO2']
            sp = np.array(sp.astype('float32')).reshape(-1)
            sp = np.nanmean(sp[Step_Index:Step_Index + self.frames_num])
            sp = sp.astype('float32')

            resp_name = 'Label/RESP.mat'
            resp_path = os.path.join(nowPath, resp_name)
            resp = scio.loadmat(resp_path)['RESP']
            resp = np.array(resp.astype('float32')).reshape(-1)
            resp = resp[Step_Index:Step_Index + self.frames_num]
            resp = (resp - np.min(resp)) / (np.max(resp) - np.min(resp))
            resp = resp.astype('float32')

            return gt, bvp, sp, resp

        elif self.dataName == 'UBFC':
            bvp_name = 'Label/BVP_Filt.mat'
            bvp_path = os.path.join(nowPath, bvp_name)
            bvp = scio.loadmat(bvp_path)['BVP']
            bvp = np.array(bvp.astype('float32')).reshape(-1)
            bvp = bvp[Step_Index:Step_Index + self.frames_num]
            bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))
            bvp = bvp.astype('float32')

            gt = utils.hr_cal(bvp)
            gt = np.array(gt)
            gt = gt.astype('float32')

        return gt, bvp

    def diff_normalize_data(self, data):
        """Calculate discrete difference in video data along the time-axis and nornamize by its standard deviation."""
        n, h, w, c = data.shape
        diffnormalized_len = n - 1
        diffnormalized_data = np.zeros((diffnormalized_len, h, w, c), dtype=np.float32)
        diffnormalized_data_padding = np.zeros((1, h, w, c), dtype=np.float32)
        for j in range(diffnormalized_len):
            diffnormalized_data[j, :, :, :] = (data[j + 1, :, :, :] - data[j, :, :, :]) / (
                    data[j + 1, :, :, :] + data[j, :, :, :] + 1e-7)
        diffnormalized_data = diffnormalized_data / np.std(diffnormalized_data)
        diffnormalized_data = np.append(diffnormalized_data, diffnormalized_data_padding, axis=0)
        diffnormalized_data[np.isnan(diffnormalized_data)] = 0
        return diffnormalized_data

    def __getitem__(self, idx):
        img_name = 'Align'
        #STMap_name = self.STMap_Name
        nowPath = os.path.join(self.root_dir, self.datalist[idx])
        temp = scio.loadmat(nowPath)
        nowPath = str(temp['Path'][0])
        #nowPath = nowPath.replace('/remote-home/hao.lu', '/home/jywang')
        Step_Index = int(temp['Step_Index'])
        # get HR value and bvp signal
        if self.dataName in ['PURE', 'VIPL', 'VV100']:
            gt, bvp, sp = self.getLabel(nowPath, Step_Index)
        elif self.dataName in ['HCW', 'V4V']:
            gt, bvp, rf, resp = self.getLabel(nowPath, Step_Index)
        elif self.dataName in ['On-Road-rPPG']:
            gt, bvp, sp, resp = self.getLabel(nowPath, Step_Index)
        else:
            gt, bvp = self.getLabel(nowPath, Step_Index)
        # get video
        Video_Path = os.path.join(nowPath, img_name)
        if self.dataName=='PURE':
            video_list =  sorted(os.listdir(Video_Path), key=lambda x: int(x.split('.')[0][5:]))
        else:
            video_list = sorted(os.listdir(Video_Path), key=lambda x: int(x.split('.')[0]))
        Max_frame = len(video_list)

        map_ori = []
        for i in video_list[Step_Index:Step_Index + self.frames_num]:
            map_ori.append(cv2.imread(os.path.join(Video_Path, i)))
            # try:
            #     map_ori.append(cv2.resize(cv2.imread(os.path.join(Video_Path, i)), (64, 64)))
            # except:
            #     map_ori.append(self.temp)
        map_ori = np.array(map_ori)
        # map_ori = self.diff_normalize_data(map_ori)
        _, Heith, With, _ = map_ori.shape

        Step_Index_aug = Step_Index
        if self.args.temporal_aug_rate > 0:
            if Step_Index + self.frames_num + 60 < Max_frame:
                if (random.uniform(0, 100) / 100.0) < self.args.temporal_aug_rate:
                    Step_Index_aug = int(random.uniform(30, 59) + Step_Index)
                    map_aug = []
                    for i in video_list[Step_Index_aug:Step_Index_aug + self.frames_num]:
                        map_aug.append(cv2.imread(os.path.join(Video_Path, i)))
                        # try:
                        #     map_aug.append(cv2.resize(cv2.imread(os.path.join(Video_Path, i)), (64, 64)))
                        # except:
                        #     map_aug.append(self.temp)
                    map_aug = np.array(map_aug)
                else:
                    map_aug = map_ori
            else:
                map_aug = map_ori
        else:
            map_aug = map_ori

        map_ori = map_ori.transpose(1, 2, 0, 3)
        map_aug = map_aug.transpose(1, 2, 0, 3)# Heith, With, T, 3

        if self.args.spatial_aug_rate > 0:
            if (random.uniform(0, 100) / 100.0) < self.args.spatial_aug_rate:
                temp_ratio = (1.0 * random.uniform(0, 100) / 100.0)
                Index = np.arange(Heith)
                if temp_ratio < 0.3:
                    Index[random.randint(0, Heith - 1)] = random.randint(0, Heith - 1)
                    Index[random.randint(0, Heith - 1)] = random.randint(0, Heith - 1)
                    map_aug = map_aug[Index]
                elif temp_ratio < 0.6:
                    Index[random.randint(0, Heith - 1)] = random.randint(0, Heith - 1)
                    Index[random.randint(0, Heith - 1)] = random.randint(0, Heith - 1)
                    Index[random.randint(0, Heith - 1)] = random.randint(0, Heith - 1)
                    Index[random.randint(0, Heith - 1)] = random.randint(0, Heith - 1)
                    map_aug = map_aug[Index]
                elif temp_ratio < 0.9:
                    np.random.shuffle(Index[random.randint(0, Heith - 1):random.randint(0, Heith - 1)])
                    map_aug = map_aug[Index]
                else:
                    np.random.shuffle(Index)
                    map_aug = map_aug[Index]
                # map_aug = map_aug.transpose(3, 2, 0, 1)

        for c in range(map_ori.shape[3]):
            for r in range(map_ori.shape[0]):
                for h in range(map_ori.shape[1]):
                    map_ori[r, h, :, c] = 255 * ((map_ori[r, h, :, c] - np.min(map_ori[r, h, :, c])) / \
                                                 (0.00001 + np.max(map_ori[r, h, :, c]) - np.min(map_ori[r, h, :, c])))

        for c in range(map_aug.shape[3]):
            for r in range(map_aug.shape[0]):
                for h in range(map_ori.shape[1]):
                    map_aug[r, h, :, c] = 255 * ((map_aug[r, h, :, c] - np.min(map_aug[r, h, :, c])) / \
                                                 (0.00001 + np.max(map_aug[r, h, :, c]) - np.min(map_aug[r, h, :, c])))

        if self.dataName in ['PURE', 'VIPL', 'VV100']:
            gt_aug, bvp_aug, sp_aug = self.getLabel(nowPath, Step_Index_aug)
        elif self.dataName in ['HCW', 'V4V']:
            gt_aug, bvp_aug, rf_aug, resp_aug = self.getLabel(nowPath, Step_Index_aug)
        elif self.dataName in ['On-Road-rPPG']:
            gt_aug, bvp_aug, sp_aug, resp_aug = self.getLabel(nowPath, Step_Index_aug)
        else:
            gt_aug, bvp_aug = self.getLabel(nowPath, Step_Index_aug)
        # gt_aug, bvp_aug = self.getLabel(nowPath, Step_Index_aug)


        #map_ori = map_ori.transpose(2, 3, 0, 1)
        #map_ori = utils.diff_normalize_data(map_ori)
        #map_aug = map_aug.transpose(2, 3, 0, 1)
        #map_aug = utils.diff_normalize_data(map_aug)
        map_ori = map_ori.transpose(3, 2, 0, 1)
        map_aug = map_aug.transpose(3, 2, 0, 1)

        if self.dataName in ['PURE', 'VIPL', 'VV100']:
            return (map_ori, bvp, gt, sp, 0, bvp, map_aug, bvp_aug, gt_aug, sp_aug, 0, bvp, self.domain_label)
        elif self.dataName in ['HCW', 'V4V']:
            return (map_ori, bvp, gt, 0, rf, resp, map_aug, bvp_aug, gt_aug, 0, rf_aug, resp_aug, self.domain_label)
        elif self.dataName in ['On-Road-rPPG']:
            return (map_ori, bvp, gt, sp, rf, resp, map_aug, bvp_aug, gt_aug, sp_aug, rf, resp_aug, self.domain_label)
        else:
            return (map_ori, bvp, gt, 0, 0, 0, map_aug, bvp_aug, gt_aug, 0, 0, 0, self.domain_label)


class Data_Landmark(Dataset):
    def __init__(self, root_dir, dataName, STMap, frames_num, args, transform=None, domain_label=None, datalist=None):
        self.root_dir = root_dir
        self.dataName = dataName
        self.STMap_Name = STMap
        self.frames_num = int(frames_num)
        if datalist is not None:
            self.datalist = datalist
        else:
            self.datalist = os.listdir(root_dir)
        self.datalist = sorted(self.datalist)
        self.num = len(self.datalist)
        self.domain_label = domain_label
        self.transform = transform
        self.args = args

    def __len__(self):
        return self.num

    def getLabel(self, nowPath, Step_Index):
        # 读取每个数据集的心率值和BVP信号
        if self.dataName == 'VV100':
            bvp_name = 'Label/BVP_Filt.mat'
            bvp_path = os.path.join(nowPath, bvp_name)
            bvp = scio.loadmat(bvp_path)['BVP']
            bvp = np.array(bvp.astype('float32')).reshape(-1)
            bvp = bvp[Step_Index:Step_Index + self.frames_num]
            bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))
            bvp = bvp.astype('float32')

            gt_name = 'Label/HR.mat'
            gt_path = os.path.join(nowPath, gt_name)
            gt = scio.loadmat(gt_path)['HR']
            gt = np.array(gt.astype('float32')).reshape(-1)
            gt = gt[int(Step_Index / 10)]
            gt = gt.astype('float32').reshape(-1)

            sp_name = 'Label/SPO2.mat'
            sp_path = os.path.join(nowPath, sp_name)
            sp = scio.loadmat(sp_path)['HR']  # wrong label name but correct value
            sp = np.array(sp.astype('float32')).reshape(-1)
            sp = np.nanmean(sp[Step_Index:Step_Index + self.frames_num])
            sp = sp.astype('float32')
            return gt, bvp, sp

        elif self.dataName == 'MMPD':
            bvp_name = 'Label/BVP_Filt.mat'
            bvp_path = os.path.join(nowPath, bvp_name)
            bvp = scio.loadmat(bvp_path)['BVP']
            bvp = np.array(bvp.astype('float32')).reshape(-1)
            bvp = bvp[Step_Index:Step_Index + self.frames_num]
            bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))
            bvp = bvp.astype('float32')

            gt, _, _ = utils.hr_fft(bvp, fs=30, harmonics_removal=True)
            gt = np.array(gt).reshape(-1)
            gt = gt.astype('float32')


        elif self.dataName == 'BUAA':
            bvp_name = 'Label/BVP.mat'
            bvp_path = os.path.join(nowPath, bvp_name)
            bvp = scio.loadmat(bvp_path)['BVP']
            bvp = np.array(bvp.astype('float32')).reshape(-1)
            bvp = bvp[Step_Index:Step_Index + self.frames_num]
            bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))
            bvp = bvp.astype('float32')

            gt_name = 'Label/HR_256.mat'
            gt_path = os.path.join(nowPath, gt_name)
            gt = scio.loadmat(gt_path)['HR']
            gt = np.array(gt.astype('float32')).reshape(-1)
            gt = gt[int(Step_Index / 10)]
            gt = gt.astype('float32').reshape(-1)

        elif self.dataName == 'VIPL':
            bvp_name = 'Label_CSI/BVP_Filt.mat'
            bvp_path = os.path.join(nowPath, bvp_name)
            bvp = scio.loadmat(bvp_path)['BVP']
            bvp = np.array(bvp.astype('float32')).reshape(-1)
            bvp = bvp[Step_Index:Step_Index + self.frames_num]
            bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))
            bvp = bvp.astype('float32')

            gt_name = 'Label_CSI/HR.mat'
            gt_path = os.path.join(nowPath, gt_name)
            gt = scio.loadmat(gt_path)['HR']
            gt = np.array(gt.astype('float32')).reshape(-1)
            gt = np.nanmean(gt[Step_Index:Step_Index + self.frames_num])
            gt = gt.astype('float32').reshape(-1)

            sp_name = 'Label_CSI/SPO2.mat'
            sp_path = os.path.join(nowPath, sp_name)
            sp = scio.loadmat(sp_path)['SPO2']
            sp = np.array(sp.astype('float32')).reshape(-1)
            sp = np.nanmean(sp[Step_Index:Step_Index + self.frames_num])
            sp = sp.astype('float32')

            return gt, bvp, sp

        elif self.dataName == 'V4V':
            gt_name = 'Label/HR.mat'
            gt_path = os.path.join(nowPath, gt_name)
            gt = scio.loadmat(gt_path)['HR']
            gt = np.array(gt.astype('float32')).reshape(-1)
            gt = np.nanmean(gt[Step_Index:Step_Index + self.frames_num])
            gt = gt.astype('float32').reshape(-1)
            bvp = np.array(0.0)
            bvp = bvp.astype('float32')

            rf_name = 'Label/RF.mat'
            rf_path = os.path.join(nowPath, rf_name)
            rf = scio.loadmat(rf_path)['RF']
            rf = np.array(rf.astype('float32')).reshape(-1)
            rf = np.nanmean(rf[Step_Index:Step_Index + self.frames_num])
            rf = rf.astype('float32')

            resp = np.array(0.0)
            resp = resp.astype('float32')

            return gt, bvp, rf, resp

        elif self.dataName == 'HCW':
            bvp_name = 'Label/BVP_Filt.mat'
            bvp_path = os.path.join(nowPath, bvp_name)
            bvp = scio.loadmat(bvp_path)['BVP']
            bvp = np.array(bvp.astype('float32')).reshape(-1)
            bvp = bvp[Step_Index:Step_Index + self.frames_num]
            bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))
            bvp = bvp.astype('float32')

            gt_name = 'Label/HR_Filt.mat'
            gt_path = os.path.join(nowPath, gt_name)
            gt = scio.loadmat(gt_path)['HR']
            gt = np.array(gt.astype('float32')).reshape(-1)
            gt = gt[Step_Index]  # np.nanmean(gt[Step_Index:Step_Index + self.frames_num])
            gt = gt.astype('float32')
            '''
            gt, _, _ = utils.hr_fft(bvp, fs=30, harmonics_removal=True)
            gt = np.array(gt).reshape(-1)
            gt = gt.astype('float32')'''

            rf_name = 'Label/RF_Filt.mat'
            rf_path = os.path.join(nowPath, rf_name)
            rf = scio.loadmat(rf_path)['RF']
            rf = np.array(rf.astype('float32')).reshape(-1)
            rf = rf[Step_Index]  # np.nanmean(rf[Step_Index:Step_Index + self.frames_num])
            rf = rf.astype('float32')

            resp_name = 'Label/RESP_Filt.mat'
            resp_path = os.path.join(nowPath, resp_name)
            resp = scio.loadmat(resp_path)['RESP']
            resp = np.array(resp.astype('float32')).reshape(-1)
            resp = resp[Step_Index:Step_Index + self.frames_num]
            resp = (resp - np.min(resp)) / (np.max(resp) - np.min(resp))
            resp = resp.astype('float32')

            return gt, bvp, rf, resp

        elif self.dataName == 'PURE':
            bvp_name = 'Label/BVP.mat'
            bvp_path = os.path.join(nowPath, bvp_name)
            bvp = scio.loadmat(bvp_path)['BVP']
            bvp = np.array(bvp.astype('float32')).reshape(-1)
            bvp = bvp[Step_Index:Step_Index + self.frames_num]
            bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))
            bvp = bvp.astype('float32')

            gt_name = 'Label/HR.mat'
            gt_path = os.path.join(nowPath, gt_name)
            gt = scio.loadmat(gt_path)['HR']
            gt = np.array(gt.astype('float32')).reshape(-1)
            gt = np.nanmean(gt[Step_Index:Step_Index + self.frames_num])
            gt = gt.astype('float32').reshape(-1)

            sp_name = 'Label/SPO2.mat'
            sp_path = os.path.join(nowPath, sp_name)
            sp = scio.loadmat(sp_path)['SPO2']
            sp = np.array(sp.astype('float32')).reshape(-1)
            sp = np.nanmean(sp[Step_Index:Step_Index + self.frames_num])
            sp = sp.astype('float32')
            return gt, bvp, sp

        elif self.dataName == 'UBFC':
            bvp_name = 'Label/BVP.mat'
            bvp_path = os.path.join(nowPath, bvp_name)
            bvp = scio.loadmat(bvp_path)['BVP']
            bvp = np.array(bvp.astype('float32')).reshape(-1)
            bvp = bvp[Step_Index:Step_Index + self.frames_num]
            bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))
            bvp = bvp.astype('float32')

            gt_name = 'Label/HR.mat'
            gt_path = os.path.join(nowPath, gt_name)
            gt = scio.loadmat(gt_path)['HR']
            gt = np.array(gt.astype('float32')).reshape(-1)
            gt = np.nanmean(gt[Step_Index:Step_Index + self.frames_num])
            gt = gt.astype('float32').reshape(-1)

        return gt, bvp

    def process_lmk(self, landmarks):
        delta_x = np.diff(landmarks[:, :, 0], axis=0)
        delta_y = np.diff(landmarks[:, :, 1], axis=0)
        magnitude = np.sqrt(delta_x ** 2 + delta_y ** 2)

        # Aggregate signal across selected key points (e.g., average)
        aggregated_signal = np.mean(magnitude, axis=1)

        # Apply bandpass filter
        fs = 30  # Example frame rate
        lowcut = 0.1
        highcut = 0.5
        b, a = butter(4, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype='band')
        filtered_signal = filtfilt(b, a, aggregated_signal)

        # Analyze frequency content
        frequencies = np.fft.fftfreq(len(filtered_signal), d=1 / fs)
        fft_values = np.fft.fft(filtered_signal)

        # Identify dominant frequency
        breathing_frequency = frequencies[np.argmax(np.abs(fft_values))]

        return breathing_frequency*60

    def __getitem__(self, idx):
        img_name = 'Label/RGB_lmk.csv'
        # STMap_name = self.STMap_Name
        nowPath = os.path.join(self.root_dir, self.datalist[idx])
        temp = scio.loadmat(nowPath)
        nowPath = str(temp['Path'][0])
        nowPath = nowPath.replace('/remote-home/hao.lu', '/home/jywang')
        Step_Index = int(temp['Step_Index'])
        # get HR value and bvp signal
        if self.dataName in ['PURE', 'VIPL', 'VV100']:
            gt, bvp, sp = self.getLabel(nowPath, Step_Index)
        elif self.dataName in ['HCW', 'V4V']:
            gt, bvp, rf, resp = self.getLabel(nowPath, Step_Index)
        else:
            gt, bvp = self.getLabel(nowPath, Step_Index)
        # get video
        lmk_Path = os.path.join(nowPath, img_name)
        lmk = pd.read_csv(lmk_Path)

        Max_frame = lmk.shape[0]

        if Step_Index + self.frames_num<=Max_frame:
            map_ori = lmk.values[Step_Index:Step_Index + self.frames_num, :]
        else:
            map_ori = lmk.values[Step_Index:, :]
            temp = map_ori[-1, :]
            map_ori = np.vstack((map_ori, temp.reshape(1, -1)))
        map_ori = map_ori.reshape(self.frames_num, 68, 2) # [T, 68, 2]


        # map_ori = self.diff_normalize_data(map_ori)
        _, Width, _ = map_ori.shape
        '''
        Step_Index_aug = Step_Index
        if self.args.temporal_aug_rate > 0:
            if Step_Index + self.frames_num + 60 < Max_frame:
                if (random.uniform(0, 100) / 100.0) < self.args.temporal_aug_rate:
                    Step_Index_aug = int(random.uniform(30, 59) + Step_Index)
                    map_aug = lmk.values[Step_Index_aug:Step_Index_aug + self.frames_num, :]
                    map_aug = map_aug.reshape(self.frames_num, 68, 2)
                else:
                    map_aug = map_ori
            else:
                map_aug = map_ori
        else:
            map_aug = map_ori

        map_ori = map_ori.transpose(1, 0, 2)
        #map_aug = map_aug.transpose(1, 0, 2)  # With, T, 2

        if self.args.spatial_aug_rate > 0:
            if (random.uniform(0, 100) / 100.0) < self.args.spatial_aug_rate:
                temp_ratio = (1.0 * random.uniform(0, 100) / 100.0)
                Index = np.arange(Width)
                if temp_ratio < 0.3:
                    Index[random.randint(0, Width - 1)] = random.randint(0, Width - 1)
                    Index[random.randint(0, Width - 1)] = random.randint(0, Width - 1)
                    map_aug = map_aug[Index]
                elif temp_ratio < 0.6:
                    Index[random.randint(0, Width - 1)] = random.randint(0, Width - 1)
                    Index[random.randint(0, Width - 1)] = random.randint(0, Width - 1)
                    Index[random.randint(0, Width - 1)] = random.randint(0, Width - 1)
                    Index[random.randint(0, Width - 1)] = random.randint(0, Width - 1)
                    map_aug = map_aug[Index]
                elif temp_ratio < 0.9:
                    np.random.shuffle(Index[random.randint(0, Width - 1):random.randint(0, Width - 1)])
                    map_aug = map_aug[Index]
                else:
                    np.random.shuffle(Index)
                    map_aug = map_aug[Index]
                # map_aug = map_aug.transpose(3, 2, 0, 1)'''

        for c in range(map_ori.shape[2]):
            for r in range(map_ori.shape[0]):
                 map_ori[r, :, c] = (map_ori[r, :, c] - np.min(map_ori[r, :, c])) / \
                                                 (0.00001 + np.max(map_ori[r, :, c]) - np.min(map_ori[r, :, c]))
        '''
        for c in range(map_aug.shape[2]):
            for r in range(map_aug.shape[0]):
                 map_aug[r, :, c] = (map_aug[r, :, c] - np.min(map_aug[r, :, c])) / \
                                                 (0.00001 + np.max(map_aug[r, :, c]) - np.min(map_aug[r, :, c]))
        

        if self.dataName in ['PURE', 'VIPL', 'VV100']:
            gt_aug, bvp_aug, sp_aug = self.getLabel(nowPath, Step_Index_aug)
        elif self.dataName in ['HCW', 'V4V']:
            gt_aug, bvp_aug, rf_aug, resp_aug = self.getLabel(nowPath, Step_Index_aug)
        else:
            gt_aug, bvp_aug = self.getLabel(nowPath, Step_Index_aug)
        # gt_aug, bvp_aug = self.getLabel(nowPath, Step_Index_aug)
        

        #print(self.process_lmk(map_ori.transpose(1, 0, 2)))
        #print(rf_aug)

        if self.dataName in ['PURE', 'VIPL', 'VV100']:
            return (map_ori, bvp, gt, sp, 0, 0, map_aug, bvp_aug, gt_aug, sp_aug, 0, 0, self.domain_label)
        elif self.dataName in ['HCW', 'V4V']:
            return (map_ori, bvp, gt, 0, rf, resp, map_aug, bvp_aug, gt_aug, 0, rf_aug, resp_aug, self.domain_label)
        else:
            return (map_ori, bvp, gt, 0, 0, 0, map_aug, bvp_aug, gt_aug, 0, 0, 0, self.domain_label)'''
        if self.dataName in ['PURE', 'VIPL', 'VV100']:
            return (map_ori, bvp, gt, sp, 0, 0, 0, 0, 0, 0, 0, 0, self.domain_label)
        elif self.dataName in ['HCW', 'V4V']:
            return (map_ori, bvp, gt, 0, rf, resp, 0, 0, 0, 0, 0, 0, self.domain_label)
        else:
            return (map_ori, bvp, gt, 0, 0, 0, 0, 0, 0, 0, 0, 0, self.domain_label)


def CrossValidation(root_dir, fold_num=5, fold_index=0, test_percent=20):
    datalist = os.listdir(root_dir)
    # datalist = [file for file in os.listdir(root_dir) if file[2] != 'W']
    # datalist.sort(key=lambda x: int(x))
    num = len(datalist)
    fold_size = round(((num / fold_num) - 2))
    test_fold_num = int(test_percent / 100 * 5)
    train_size = num - fold_size
    test_index = datalist[fold_index * fold_size:fold_index * fold_size + fold_size * test_fold_num - 1]
    train_index = datalist[0:fold_index * fold_size] + datalist[fold_index * fold_size + fold_size * test_fold_num:]
    return train_index, test_index

# def CrossValidation(root_dir, fold_num=5, fold_index=0, test_percent=20):
#     datalist = os.listdir(root_dir)
#     # datalist.sort(key=lambda x: int(x))
#     num = len(datalist)
#     test_num = int(test_percent / 100 * num)
#     print(test_num)
#     test_index = datalist[0:test_num - 1]
#     train_index = datalist[test_num:]
#     return train_index, test_index


def getIndex(root_path, filesList, save_path, Pic_path, Step, frames_num):
    Index_path = []
    print('Now processing' + root_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for sub_file in filesList:
        if sub_file.startswith('processed'):
            continue
        now = os.path.join(root_path, sub_file)
        for subject_part in os.listdir(now):
            now_path = os.path.join(now, subject_part)
            img_path = os.path.join(now_path, 'Align')

            bvp_name = 'Label/BVP.mat'
            bvp_path = os.path.join(now_path, bvp_name)
            bvp = scio.loadmat(bvp_path)['BVP']
            bvp = np.array(bvp.astype('float32')).reshape(-1)

            # print(img_path)
            Num = len(os.listdir(img_path))
            Res = Num - frames_num - 1  # 可能是Diff数据
            Step_num = int(Res / Step)
            for i in range(Step_num):
                Step_Index = i * Step

                bvp_t = bvp[Step_Index:Step_Index + 256]
                if np.max(bvp_t) - np.min(bvp_t) == 0:
                    continue


                temp_path = sub_file + '_' + subject_part + '_' + str(1000 + i) + '_.mat'
                print(temp_path)
                scio.savemat(os.path.join(save_path, temp_path), {'Path': now_path , 'Step_Index': Step_Index})
                Index_path.append(temp_path)
    return Index_path


def calculate_respiration_rate(breathing_signal, sampling_rate=30):
    """
    Calculate the respiration rate from a breathing signal.

    :param breathing_signal: A 1-D numpy array of breathing signal data.
    :param sampling_rate: Sampling rate of the signal in Hz, default is 30Hz.
    :return: Respiration rate in breaths per minute.
    """
    peaks, _ = find_peaks(breathing_signal)
    num_of_breaths = len(peaks)
    duration_in_seconds = len(breathing_signal) / sampling_rate
    duration_in_minutes = duration_in_seconds / 60

    respiration_rate = num_of_breaths / duration_in_minutes
    return respiration_rate

# Example usage
# Replace 'your_breathing_signal_array' with your actual breathing signal data
# breathing_signal = np.array(your_breathing_signal_array)
# respiration_rate = calculate_respiration_rate(breathing_signal)
# print(f"Respiration Rate: {respiration_rate} breaths/minute")
def group_samples(root_dir, conditions, samples=None):
    """
    根据给定的实验条件对样本进行分组。

    每个样本字符串格式: sub_file + '_' + seq + '_' + i
      - sub_file: 三字符, 分别表示 车型(car), 性别(gender), 时间(time)
          * car: 'A','B','C'
          * gender: 'M','F'
          * time: 'Z','H','Y','W'
      - seq: 两字符, 分别表示 路况(difficulty), 说话状态(speech)
          * difficulty: 'A','B','C'
          * speech: '1' (不说话), '2' (说话)
      - i: 索引或其他信息，可忽略

    :param samples: 样本列表
    :param conditions: 实验条件列表, 最多两个. 可选值:
                       'car'       (车型),
                       'gender'    (性别),
                       'time'      (时段),
                       'difficulty'(路况),
                       'speech'    (说话状态)
    :return: 字典, 键为由条件值拼接的分组名, 值为该组对应的样本列表
    """
    if samples is None:
        samples = os.listdir(root_dir)
    # 支持的条件
    allowed = {'car', 'gender', 'time', 'difficulty', 'speech'}
    if len(conditions) > 2:
        raise ValueError("最多只能指定两个实验条件")
    for cond in conditions:
        if cond not in allowed:
            raise ValueError(f"未知的实验条件: {cond}")

    def parse_sample(s: str) -> Dict[str, str]:
        # 拆分为 sub_file, seq, 后缀
        parts = s.split('_', 2)
        if len(parts) < 2:
            raise ValueError(f"样本格式不正确: {s}")
        sub_file, seq = parts[0], parts[1]
        if len(sub_file) != 4 or len(seq) != 2:
            raise ValueError(f"sub_file 或 seq 长度不符合要求: {s}")
        return {
            'car': sub_file[0],
            'gender': sub_file[1],
            'time': sub_file[2],
            'difficulty': seq[0],
            'speech': seq[1]
        }

    groups: Dict[str, List[str]] = {}

    for sample in samples:
        attrs = parse_sample(sample)
        # 生成分组键
        if conditions:
            key_vals = [attrs[c] for c in conditions]
            key = '_'.join(key_vals)
        else:
            key = 'all'
        groups.setdefault(key, []).append(sample)

    return groups
