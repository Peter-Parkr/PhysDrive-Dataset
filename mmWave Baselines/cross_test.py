import pandas as pd
import scipy.io as io
import torch
import torch.nn as nn
import numpy as np
import MyDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import utils
from utils import Logger, time_to_str, rr_cal
from datetime import datetime
import os
from timeit import default_timer as timer
import random
from tqdm import tqdm
import pynvml
import warnings
import MyLoss

import Model

import shutil

warnings.simplefilter('ignore')

TARGET_DOMAIN = {
                 'Physdrive': ['Physdrive']
                 }

FILEA_NAME = {
              'Physdrive': ['Physdrive', 'Physdrive', 'mmwave']
              }

if __name__ == '__main__':

    args = utils.get_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    Source_domain_Names = TARGET_DOMAIN[args.tgt]
    root_file = r'xxx' # your data address
    

    FILE_Name = FILEA_NAME[args.tgt]
    Target_name = args.tgt
    Target_fileRoot = root_file + FILE_Name[0]
    Target_saveRoot = root_file + 'MMWave_Index/' + FILE_Name[1]
    Target_map = FILE_Name[2] + '.mat'

    batch_size_num = args.batchsize
    epoch_num = args.epochs
    learning_rate = args.lr

    test_batch_size = args.batchsize
    num_workers = args.num_workers
    GPU = args.GPU

    input_form = args.form
    reTrain = args.reTrain
    frames_num = args.frames_num
    fold_num = args.fold_num
    fold_index = args.fold_index

    if args.reData == 1:
        Target_index = os.listdir(Target_fileRoot)

        Target_Indexa = MyDataset.getIndex(Target_fileRoot, Target_index, \
                                           Target_saveRoot, Target_map, 10, frames_num)

    train_list, test_list = MyDataset.CrossValidation(Target_saveRoot, fold_num=args.fold_num,
                                                      test_percent=args.test_percent)
    group_list = MyDataset.group_samples(test_list)
    del group_list['A']
    del group_list['B']
    del group_list['C']
    group_list['all'] = test_list
    for k in group_list.keys():
        print(len(group_list[k]), k)

    batch_size_num = args.batchsize
    epoch_num = args.epochs
    learning_rate = args.lr

    test_batch_size = args.batchsize
    num_workers = args.num_workers
    GPU = args.GPU

    input_form = args.form
    reTrain = args.reTrain
    frames_num = args.frames_num
    fold_num = args.fold_num
    fold_index = args.fold_index

    best_mae = 99

    print('batch num:', batch_size_num, ' epoch_num:', epoch_num, ' GPU Inedex:', GPU)
    print(' frames num:', frames_num, ' learning rate:', learning_rate, )
    print('fold num:', frames_num, ' fold index:', fold_index)

    if not os.path.exists('./Result_log'):
        os.makedirs('./Result_log')
    rPPGNet_name = 'rPPGNet_' + Target_name + 'Spatial' + str(args.spatial_aug_rate) + 'Temporal' + str(
        args.temporal_aug_rate)
    log = Logger()
    log.open('./Result_log/' + Target_name + '_' + str(reTrain) + '_' + args.m + '_intra_supervised' + str(
        args.test_percent) + '_log.txt', mode='a')
    log.write("\n----------------------------------------------- [START %s] %s\n\n" % (
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))

    pynvml.nvmlInit()
    flag = 0
    max_g = []
    spaces = []
    GPU = '10'
    for gpu in range(8):
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_Gpu = meminfo.free / 1024 / 1024 / 1024
        if free_Gpu > 35:
            flag = 1
            GPU = str(gpu)
            print("GPU:", GPU)
            print("free_Gpu:", free_Gpu)
            max_g = GPU
            break
        print("GPU:", gpu)
        print("free_Gpu:", free_Gpu)

    if args.GPU != 10 and GPU == '10':
        GPU = str(args.GPU)
    if torch.cuda.is_available():
        device = torch.device('cuda:' + GPU if torch.cuda.is_available() else 'cpu')  #
        print('on GPU ', GPU)
    else:
        print('on CPU')

    for key in group_list.keys():
        datalist = group_list[key]
        print('Test:\n')
        print('Current group:  ' + key)

        Target_db = MyDataset.Data_DG(root_dir=Target_saveRoot, dataName=Target_name, \
                                      MMRadar=Target_map, frames_num=frames_num, args=args, domain_label=5,
                                      datalist=datalist)

        tgt_loader = DataLoader(Target_db, batch_size=batch_size_num, shuffle=False, num_workers=0)

        tgt_iter = iter(tgt_loader)
        tgt_iter_per_epoch = len(tgt_iter)

        my_model = Model.My_model(model_name=args.m)
        if reTrain == 1:
            Net_name = 'xxxx' #your pretrained model address
            my_model = torch.load(
                Net_name,
                map_location=device)
            print('load ' + Net_name + ' right')
        my_model.to(device=device)

        HR_pr_temp = []
        HR_rel_temp = []
        ecg_ALL = []
        ecg_PR_ALL = []
        RF_pr_temp = []
        RF_rel_temp = []
        Resp_ALL = []
        Resp_PR_ALL = []
        for step, (data, ecg, resp, HR_rel, RF_rel, _, _, _, _, _) in tqdm(enumerate(tgt_loader)):
            data = Variable(data).float().to(device=device)
            Wave = Variable(ecg).float()
            Resp = Variable(resp).float()
            HR_rel = Variable(HR_rel).float()
            RF_rel = Variable(RF_rel).float()
            Wave = Wave.unsqueeze(dim=1)
            Resp = Resp.unsqueeze(dim=1)
            rand_idx = torch.randperm(data.shape[0])
            Wave_pr, Resp_pr, HR_pr, RF_pr = my_model(data)

            HR_rel_temp.extend(HR_rel.data.cpu().numpy())
            HR_pr_temp.extend(HR_pr.data.cpu().numpy())
            RF_pr_temp.extend(RF_pr.data.cpu().numpy())
            RF_rel_temp.extend(RF_rel.data.cpu().numpy())
            ecg_ALL.extend(Wave.data.cpu().numpy())
            ecg_PR_ALL.extend(Wave_pr.data.cpu().numpy())
            Resp_ALL.extend(Resp.data.cpu().numpy())
            Resp_PR_ALL.extend(Resp_pr.data.cpu().numpy())

        ME, STD, MAE, RMSE, MER, P = utils.MyEval(HR_pr_temp, HR_rel_temp)
        log.write(
            'Test Inter HR:' \
            + ' | ME:  ' + str(ME) \
            + ' | STD: ' + str(STD) \
            + ' | MAE: ' + str(MAE) \
            + ' | RMSE: ' + str(RMSE) \
            + ' | MER: ' + str(MER) \
            + ' | P ' + str(P))
        log.write('\n')

        ME, STD, MAE, RMSE, MER, P = utils.MyEval(RF_pr_temp, RF_rel_temp)
        log.write(
            'Test Inter RR:' \
            + ' | ME:  ' + str(ME) \
            + ' | STD: ' + str(STD) \
            + ' | MAE: ' + str(MAE) \
            + ' | RMSE: ' + str(RMSE) \
            + ' | MER: ' + str(MER) \
            + ' | P ' + str(P))
        log.write('\n')


        ME, STD, MAE, RMSE, MER, P = utils.MyEval_ecg_hr(ecg_PR_ALL, ecg_ALL)
        log.write(
            'Test Inter HR from ECG:'  \
            + ' | ME:  ' + str(ME) \
            + ' | STD: ' + str(STD) \
            + ' | MAE: ' + str(MAE) \
            + ' | RMSE: ' + str(RMSE) \
            + ' | MER: ' + str(MER) \
            + ' | P ' + str(P))
        log.write('\n')

        ME, STD, MAE, RMSE, MER, P = utils.MyEval_resp_rr(Resp_PR_ALL, Resp_ALL)
        log.write(
            'Test Inter RR from RESP:' \
            + ' | ME:  ' + str(ME) \
            + ' | STD: ' + str(STD) \
            + ' | MAE: ' + str(MAE) \
            + ' | RMSE: ' + str(RMSE) \
            + ' | MER: ' + str(MER) \
            + ' | P ' + str(P))
        log.write('\n')
