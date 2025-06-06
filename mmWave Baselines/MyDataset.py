# -*- coding: UTF-8 -*-
import numpy as np
import os
from torch.utils.data import Dataset
import cv2
import csv
import scipy.io as scio
import neurokit2 as nk
import torchvision.transforms.functional as transF
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from utils import rr_cal
import random
import utils
import torch
from collections import defaultdict





class Data_DG(Dataset):
    def __init__(self, root_dir, dataName, MMRadar, frames_num, args, transform=None, domain_label=None, datalist=None,
                 peoplelist=None, output_people=False):
        self.root_dir = root_dir
        self.dataName = dataName
        self.MMRadar_Name = MMRadar
        self.frames_num = int(frames_num)
        if datalist is None:
            self.datalist = os.listdir(root_dir)
            self.datalist = list(sorted(self.datalist))
        else:
            self.datalist = datalist

        self.output_people = output_people
        if output_people:
            self.peoplelist = list(peoplelist)

        self.num = len(self.datalist)
        self.domain_label = domain_label
        self.transform = transform
        self.args = args

    def __len__(self):
        return self.num

    def getLabel(self, nowPath, Step_Index):

        if self.dataName == 'Physdrive':
            ecg_name = 'ecg.mat'
            ecg_path = os.path.join(nowPath, ecg_name)
            ecg = scio.loadmat(ecg_path)['ecg']
            ecg = np.array(ecg.astype('float32')).reshape(-1)
            ecg = ecg[Step_Index:Step_Index + self.frames_num]
            ecg = (ecg - np.min(ecg)) / (np.max(ecg) - np.min(ecg))
            ecg = ecg.astype('float32')

            peaks = nk.ecg_findpeaks(ecg, sampling_rate=20, method='rodrigues2021')['ECG_R_Peaks']
            peaks -= 1
            rr_intervals = np.diff(peaks) / 20 * 1000  # 将间隔转换为毫秒
            nn_intervals = rr_intervals
            gt = 60000 / np.mean(nn_intervals)
            # gt = nk.ecg_rate(peaks=peaks, sampling_rate=20)
            # gt = np.array(gt.astype('float32')).reshape(-1)
            # gt = np.nanmean(gt)

            resp_name = 'resp.mat'
            resp_path = os.path.join(nowPath, resp_name)
            resp = scio.loadmat(resp_path)['resp']
            resp = np.array(resp.astype('float32')).reshape(-1)
            resp = resp[Step_Index:Step_Index + self.frames_num]
            resp = (resp - np.min(resp)) / (np.max(resp) - np.min(resp))
            resp = resp.astype('float32')
            try:
                rr = nk.rsp_rate(resp, sampling_rate=20, method="xcorr")
            except:
                rr = utils.rr_cal(resp, 20)
            rr = rr.astype('float32')
            rr = np.nanmean(rr)

            return ecg, resp, gt, rr

    def __getitem__(self, idx):
        MMRadar_name = self.MMRadar_Name
        nowPath = os.path.join(self.root_dir, self.datalist[idx])
        temp = scio.loadmat(nowPath)
        nowPath = str(temp['Path'][0])
        Step_Index = int(temp['Step_Index'])
        # people_i = nowPath.split('/')[-1]
        # get HR value and bvp signal
        ecg, resp, gt, rr = self.getLabel(nowPath, Step_Index)
        # get MMRadar
        MMRadar_Path = os.path.join(nowPath, MMRadar_name)
        feature_map = scio.loadmat(MMRadar_Path)['mmwave']
        #feature_map = np.array(feature_map.astype('float32'))

        MAX_frames, _,  num_doppler, num_angles, num_ranges = feature_map.shape
        # get original map
        map_ori = feature_map[Step_Index:Step_Index + self.frames_num, :, :, :, :]
        # get augmented map
        Spatial_aug_flag = 0
        Temporal_aug_flag = 0
        Step_Index_aug = Step_Index
        map_aug = map_ori

        if self.args.temporal_aug_rate > 0:
            if Step_Index + self.frames_num + 20 < MAX_frames:
                if (random.uniform(0, 100) / 100.0) < self.args.temporal_aug_rate:
                    Step_Index_aug = int(random.uniform(0, 19) + Step_Index)
                    map_aug = feature_map[Step_Index_aug:Step_Index_aug + self.frames_num, :, :, :, :]
                    Temporal_aug_flag = 1

        if (random.uniform(0, 100) / 100.0) < self.args.spatial_aug_rate:
            map_aug = map_aug.reshape(self.frames_num, 2, num_doppler, 4, 4, 2, 4)
            map_aug = map_aug.transpose(0, 1, 2, 3, 5, 4, 6)

            for i in range(self.frames_num):
                for j in range(num_doppler):
                    for k in range(2):
                        patches = map_aug[i, j, k]
                        patches_flat = patches.reshape(-1, 4, 4)
                        np.random.shuffle(patches_flat)
                        patches_shuffled = patches_flat.reshape(4, 2, 4, 4)
                        map_aug[i, j, k] = patches_shuffled

            map_aug = map_aug.transpose(0, 1, 2, 3, 5, 4, 6).reshape(self.frames_num, 2, num_doppler, num_angles, num_ranges)

        if ((Spatial_aug_flag == 0) and (Temporal_aug_flag == 0)):
            map_aug = map_ori

        ecg_aug, resp_aug, gt_aug, rr_aug = self.getLabel(nowPath, Step_Index_aug)


        return (map_ori, ecg, resp, gt, rr, map_aug, ecg_aug, resp_aug, gt_aug, rr_aug)


def group_samples(data_list):
    result = {
        'A': [], 'B': [], 'C': [],
        'Easy_Road': [], 'Hard_Road': [], 'Medium_Road': [],
        'quiet': [], 'speak': []
    }

    user_groups = defaultdict(lambda: {'car_type': None, 'data': []})

    for s in data_list:
        try:
            parts = s.split('_')
            if len(parts) != 4 or len(parts[0]) != 4:
                continue

            user_id = parts[0]  
            car_type = user_id[0] 
            part2 = int(parts[1]) 

            if user_groups[user_id]['car_type'] is None:
                user_groups[user_id]['car_type'] = car_type
            elif user_groups[user_id]['car_type'] != car_type:
                continue  

            user_groups[user_id]['data'].append((part2, s))

        except (ValueError, IndexError):
            continue

    mapping = [
        ('Easy_Road', 'quiet'),  
        ('Easy_Road', 'speak'), 
        ('Hard_Road', 'quiet'),  
        ('Medium_Road', 'speak'), 
        ('Medium_Road', 'quiet'), 
        ('Hard_Road', 'speak')  
    ]

    for user_id, user_info in user_groups.items():
        car_type = user_info['car_type']
        if car_type not in ['A', 'B', 'C']:
            continue

        sorted_data = sorted(user_info['data'], key=lambda x: x[0])
        n = len(sorted_data)
        segments = []
        start = 0

        base_size, remainder = divmod(n, 6)
        for i in range(6):
            end = start + base_size + (1 if i < remainder else 0)
            seg = [item[1] for item in sorted_data[start:end]]
            segments.append(seg)
            start = end

        for seg_idx in range(6):
            road, action = mapping[seg_idx]
            for s in segments[seg_idx]:
                result[car_type].append(s)
                result[road].append(s)
                result[action].append(s)

    return result


def CrossValidation(root_dir, fold_num=5, fold_index=0, test_percent=20):
    datalist = os.listdir(root_dir)
    # datalist.sort(key=lambda x: int(x))
    num = len(datalist)
    fold_size = round(((num / fold_num) - 2))
    test_fold_num = int(test_percent / 100 * 5)
    train_size = num - fold_size
    test_index = datalist[fold_index * fold_size:fold_index * fold_size + fold_size * test_fold_num - 1]
    train_index = datalist[0:fold_index * fold_size] + datalist[fold_index * fold_size + fold_size * test_fold_num:]
    return train_index, test_index


def getIndex(root_path, filesList, save_path, wave_file, Step, frames_num):
    Index_path = []
    print('Now processing' + root_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for sub_file in filesList:
        now = os.path.join(root_path, sub_file)
        wave_path = os.path.join(now, wave_file)
        temp = scio.loadmat(wave_path)['mmwave']
        Num = temp.shape[0]

