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
        
        if self.dataName == 'PURE':
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

        elif self.dataName == 'PhysDrive':
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
        nowPath = os.path.join(self.root_dir, self.datalist[idx])
        temp = scio.loadmat(nowPath)
        nowPath = str(temp['Path'][0])
        Step_Index = int(temp['Step_Index'])
        # get HR value and bvp signal
        if self.dataName in ['PURE', 'VIPL', 'VV100']:
            gt, bvp, sp = self.getLabel(nowPath, Step_Index)
        elif self.dataName in ['HCW', 'V4V']:
            gt, bvp, rf, resp = self.getLabel(nowPath, Step_Index)
        elif self.dataName in ['PhysDrive']:
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
        elif self.dataName in ['PhysDrive']:
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
        elif self.dataName in ['PhysDrive']:
            return (map_ori, bvp, gt, sp, rf, resp, map_aug, bvp_aug, gt_aug, sp_aug, rf, resp_aug, self.domain_label)
        else:
            return (map_ori, bvp, gt, 0, 0, 0, map_aug, bvp_aug, gt_aug, 0, 0, 0, self.domain_label)



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



def group_samples(root_dir, conditions, samples=None):
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
