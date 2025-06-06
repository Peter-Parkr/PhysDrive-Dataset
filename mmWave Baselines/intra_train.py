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
    root_file = r'xxxx' #your data address


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

    source_db_0 = MyDataset.Data_DG(root_dir=Target_saveRoot, dataName=Target_name, \
                                    MMRadar=Target_map, frames_num=frames_num, args=args, domain_label=0,
                                    datalist=train_list)
    Target_db = MyDataset.Data_DG(root_dir=Target_saveRoot, dataName=Target_name, \
                                  MMRadar=Target_map, frames_num=frames_num, args=args, domain_label=5,
                                  datalist=test_list)

    src_loader_0 = DataLoader(source_db_0, batch_size=batch_size_num, shuffle=True, num_workers=num_workers)
    tgt_loader = DataLoader(Target_db, batch_size=batch_size_num, shuffle=False, num_workers=0)

    my_model = Model.My_model(model_name=args.m)
    
    my_model.to(device=device)



    optimizer_rPPG = torch.optim.AdamW(my_model.parameters(), lr=learning_rate)
    warmup_iters = args.warmup_iters if hasattr(args, 'warmup_iters') else 1000  
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer_rPPG,
        lr_lambda=lambda current_step: float(current_step) / warmup_iters if current_step < warmup_iters else 1.0
    )

    loss_func_ECG = [MyLoss.Cos_loss().to(device), nn.SmoothL1Loss().to(device)]
    loss_func_RESP = [MyLoss.Cos_loss().to(device), nn.SmoothL1Loss().to(device)]
    loss_func_HR = nn.SmoothL1Loss().to(device)
    loss_func_RR = nn.SmoothL1Loss().to(device)

    src_iter_0 = src_loader_0.__iter__()
    src_iter_per_epoch_0 = len(src_iter_0)

    tgt_iter = iter(tgt_loader)
    tgt_iter_per_epoch = len(tgt_iter)

    max_iter = args.max_iter
    start = timer()
    loss_res = {'ecg': [], 'hr': [], 'rr': [], 'resp': [], 'all': []}

    eval_ecg_hr = {'MAE': [], 'RMSE': [], 'MER': [], 'P': []}
    eval_resp_rr = {'MAE': [], 'RMSE': [], 'MER': [], 'P': []}
    eval_hr = {'MAE': [], 'RMSE': [], 'MER': [], 'P': []}
    eval_rr = {'MAE': [], 'RMSE': [], 'MER': [], 'P': []}

    with tqdm(range(max_iter + 1)) as it:
        for iter_num in it:
            my_model.train()
            if (iter_num % src_iter_per_epoch_0 == 0):
                src_iter_0 = src_loader_0.__iter__()

            ######### data prepare #########
            data0, ecg0, resp0, HR_rel0, rf_rel0, data_aug0, ecg_aug0, resp_aug0, HR_rel_aug0, rf_rel_aug0 = src_iter_0.__next__()

            data0 = Variable(data0).float().to(device=device)
            ecg0 = Variable(ecg0).float().to(device=device).unsqueeze(dim=1)
            resp0 = Variable(resp0).float().to(device=device).unsqueeze(dim=1)
            HR_rel0 = Variable(HR_rel0).float().to(device=device)
            rf_rel0 = Variable(rf_rel0).float().to(device=device)

            data_aug0 = Variable(data_aug0).float().to(device=device)
            ecg_aug0 = Variable(ecg_aug0).float().to(device=device).unsqueeze(dim=1)
            resp_aug0 = Variable(resp_aug0).float().to(device=device).unsqueeze(dim=1)
            HR_rel_aug0 = Variable(HR_rel_aug0).float().to(device=device)
            rf_rel_aug0 = Variable(rf_rel_aug0).float().to(device=device)

            optimizer_rPPG.zero_grad()

            input = data0
            input_aug = data_aug0

            ecg_pre_zip, resp_pre_zip, HR_pr_zip, rf_pre_zip = my_model(input)

            ecg_pre_aug_zip, resp_pre_aug_zip, HR_pr_aug_zip, rf_pre_aug_zip = my_model(input_aug)

            rppg_loss_0, loss_res = MyLoss.get_loss(ecg_pre_zip, resp_pre_zip, HR_pr_zip, rf_pre_zip, ecg0,
                                                    resp0, HR_rel0, rf_rel0, Target_name, \
                                                    loss_func_ECG, loss_func_RESP, loss_func_HR, loss_func_RR, args,
                                                    iter_num,
                                                    loss_res=loss_res)

            rppg_loss_aug_0, _ = MyLoss.get_loss(ecg_pre_aug_zip, resp_pre_aug_zip, HR_pr_aug_zip,
                                                 rf_pre_aug_zip, ecg_aug0, resp_aug0, HR_rel_aug0,
                                                 rf_rel_aug0,
                                                 Target_name, \
                                                 loss_func_ECG, loss_func_RESP, loss_func_HR, loss_func_RR, args,
                                                 iter_num)

            # if torch.sum(torch.isnan(rppg_loss_0)) > 0 or torch.sum(torch.isnan(rppg_loss_aug_0)) > 0:
            # continue

            rppg_loss = rppg_loss_0 + rppg_loss_aug_0

            # k = 2.0 / (1.0 + np.exp(-10.0 * iter_num / args.max_iter)) - 1.0
            loss_all = rppg_loss  # + 0.1 * k * loss_TA + 0.001 * k * loss_CM + 0.01 * k * loss_DM
            loss_res['all'].append(loss_all.item())

            if torch.sum(torch.isnan(rppg_loss)) > 0:
                print('Nan')
                continue
            else:
                rppg_loss.backward()
                optimizer_rPPG.step()
                scheduler.step()

                it.set_postfix(
                    ordered_dict={
                        "Train Inter": iter_num,
                        "overall loss": rppg_loss.data.cpu().numpy(),
                    },
                    refresh=False,
                )

            log.write(
                'Train Inter:' + str(iter_num) \
                + ' | Overall Loss:  ' + str(loss_all.data.cpu().numpy()) \
                + ' |' + time_to_str(timer() - start, 'min'))
            log.write('\n')

            if (iter_num > 0) and (iter_num % 1000 == 0):
      
                print('Test:\n')

                utils.loss_visual(loss_res,
                                  './visuals/loss_visual/main_task_loss_key_' + args.m + '_' + args.tgt + '_' + str(
                                      args.lr) + '_intra_supervised.png')

                my_model.eval()
                loss_mean = []
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
                print(Target_name)
                log.write(
                    'Test Inter HR:' + str(iter_num) \
                    + ' | ME:  ' + str(ME) \
                    + ' | STD: ' + str(STD) \
                    + ' | MAE: ' + str(MAE) \
                    + ' | RMSE: ' + str(RMSE) \
                    + ' | MER: ' + str(MER) \
                    + ' | P ' + str(P))
                log.write('\n')
                eval_hr['MAE'].append(MAE)
                eval_hr['RMSE'].append(RMSE)
                eval_hr['MER'].append(MER)
                eval_hr['P'].append(P)
                ME, STD, MAE, RMSE, MER, P = utils.MyEval(RF_pr_temp, RF_rel_temp)
                log.write(
                    'Test Inter RR:' + str(iter_num) \
                    + ' | ME:  ' + str(ME) \
                    + ' | STD: ' + str(STD) \
                    + ' | MAE: ' + str(MAE) \
                    + ' | RMSE: ' + str(RMSE) \
                    + ' | MER: ' + str(MER) \
                    + ' | P ' + str(P))
                log.write('\n')
                eval_rr['MAE'].append(MAE)
                eval_rr['RMSE'].append(RMSE)
                eval_rr['MER'].append(MER)
                eval_rr['P'].append(P)

                ME, STD, MAE, RMSE, MER, P = utils.MyEval_ecg_hr(ecg_PR_ALL, ecg_ALL)
                log.write(
                    'Test Inter HR from ECG:' + str(iter_num) \
                    + ' | ME:  ' + str(ME) \
                    + ' | STD: ' + str(STD) \
                    + ' | MAE: ' + str(MAE) \
                    + ' | RMSE: ' + str(RMSE) \
                    + ' | MER: ' + str(MER) \
                    + ' | P ' + str(P))
                log.write('\n')
                eval_ecg_hr['MAE'].append(MAE)
                eval_ecg_hr['RMSE'].append(RMSE)
                eval_ecg_hr['MER'].append(MER)
                eval_ecg_hr['P'].append(P)
                ME, STD, MAE, RMSE, MER, P = utils.MyEval_resp_rr(Resp_PR_ALL, Resp_ALL)
                log.write(
                    'Test Inter RR from RESP:' + str(iter_num) \
                    + ' | ME:  ' + str(ME) \
                    + ' | STD: ' + str(STD) \
                    + ' | MAE: ' + str(MAE) \
                    + ' | RMSE: ' + str(RMSE) \
                    + ' | MER: ' + str(MER) \
                    + ' | P ' + str(P))
                log.write('\n')
                eval_resp_rr['MAE'].append(MAE)
                eval_resp_rr['RMSE'].append(RMSE)
                eval_resp_rr['MER'].append(MER)
                eval_resp_rr['P'].append(P)

                eval_hr_save = pd.DataFrame(eval_hr)
                eval_hr_save.to_csv('./visuals/result_visual/' + Target_name + '_' + str(
                    args.lr) + '_' + str(
                    reTrain) + '_' + args.m + '_intra_supervised' + str(args.test_percent) + '_HR.csv')
                eval_ecg_hr_save = pd.DataFrame(eval_ecg_hr)
                eval_ecg_hr_save.to_csv('./visuals/result_visual/' + Target_name + '_' + str(
                    args.lr) + '_' + str(
                    reTrain) + '_' + args.m + '_intra_supervised' + str(args.test_percent) + '_ECG_HR.csv')
                eval_resp_rr_save = pd.DataFrame(eval_resp_rr)
                eval_resp_rr_save.to_csv('./visuals/result_visual/' + Target_name + '_' + str(
                    args.lr) + '_' + str(
                    reTrain) + '_' + args.m + '_intra_supervised' + str(args.test_percent) + '_Resp_RR.csv')
                eval_rr_save = pd.DataFrame(eval_rr)
                eval_rr_save.to_csv('./visuals/result_visual/' + Target_name + '_' + str(
                    args.lr) + '_' + str(
                    reTrain) + '_' + args.m + '_intra_supervised' + str(args.test_percent) + '_RR.csv')

                if not os.path.exists('./Result_Model'):
                    os.makedirs('./Result_Model')
                
                torch.save(my_model,
                           './Result_Model/' + Target_name + '_' + str(iter_num) + '_' + str(
                               reTrain) + '_' + args.m + '_intra_supervised' + str(args.test_percent))
                print('saveModel As ' + Target_name + '_' + str(iter_num) + '_' + str(
                    reTrain) + '_' + args.m + '_intra_supervised' + str(args.test_percent))
