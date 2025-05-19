import pandas as pd
import scipy.io as io
import torch
import torch.nn as nn
import numpy as np
import MyDataset
import MyLoss
from Video_Baselines.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from Video_Baselines.MTTS_CAN import MTTS_CAN
from Video_Baselines.Physformer import ViT_ST_ST_Compact3_TDC_gra_sharp
from Video_Baselines.RhythmFormer import RhythmFormer
from Video_Baselines.DeepPhys import DeepPhys
from Video_Baselines.SiNC import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import random
# from thop import profile
# from basic_module import *
import utils
from datetime import datetime
import os
from utils import Logger, time_to_str, rr_cal
from timeit import default_timer as timer
import random
from tqdm import tqdm
import pynvml
from losses import loss_init
import warnings
# from CHROM import build_rPPG_model
from Green import build_rPPG_model

# from POS import build_rPPG_model
# from ICA import build_rPPG_model

warnings.simplefilter('ignore')

TARGET_DOMAIN = {'VIPL': ['VIPL'], \
                 'V4V': ['V4V'], \
                 'PURE': ['PURE'], \
                 'BUAA': ['BUAA'], \
                 'UBFC': ['UBFC'], \
                 'HCW': ['HCW'],
                 'VV100': ['VV100'],
                 'MMPD': ['MMPD']}

FILEA_NAME = {'VIPL': ['VIPL', 'VIPL', 'STMap_RGB_Align_CSI'], \
              'V4V': ['V4V', 'V4V', 'STMap_RGB'], \
              'PURE': ['PURE', 'PURE', 'STMap'], \
              'BUAA': ['BUAA', 'BUAA', 'STMap_RGB'], \
              'UBFC': ['UBFC', 'UBFC', 'STMap'], \
              'HCW': ['HCW', 'HCW', 'STMap_RGB'],
              'VV100': ['VV100', 'VV100', 'STMap_RGB'],
              'MMPD': ['MMPD', 'MMPD', 'STMap_RGB']
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
    root_file = r'/remote-home/hao.lu/Data/Video_all/'
    # 参数

    # 图片参数
    input_form = args.form
    reTrain = args.reTrain
    frames_num = args.frames_num
    fold_num = args.fold_num
    fold_index = args.fold_index

    best_mae = 99

    FILE_Name = FILEA_NAME[args.tgt]
    Target_name = args.tgt
    Target_fileRoot = root_file + FILE_Name[0]
    Target_fileRoot = Target_fileRoot.replace('STMap/HCW', 'GPT-Chat/ChatGPT')
    Target_saveRoot = root_file + 'STMap_Index/' + FILE_Name[1]
    Target_saveRoot = Target_saveRoot.replace('STMap/STMap_Index/HCW', 'GPT-Chat/STMap_Index/HCW')
    Target_map = FILE_Name[2] + '.png'

    if args.reData == 1:
        Target_index = os.listdir(Target_fileRoot)

        Target_Indexa = MyDataset.getIndex(Target_fileRoot, Target_index, \
                                           Target_saveRoot, Target_map, 30, frames_num)
    # group_list = MyDataset.group_samples(Target_saveRoot, ['time'])
    # group_list.update(MyDataset.group_samples(Target_saveRoot, ['speech']))
    #
    # temp = []
    # for k in group_list.keys():
    #     group_list[k] = group_list[k][:2000]
    #     temp += group_list[k]
    # group_list['all'] = list(set(temp))
    # # group_list['all'] = os.listdir(Target_saveRoot)

    # 训练参数
    batch_size_num = args.batchsize
    epoch_num = args.epochs
    learning_rate = args.lr

    test_batch_size = args.batchsize
    num_workers = args.num_workers
    GPU = args.GPU

    print('batch num:', batch_size_num, ' epoch_num:', epoch_num, ' GPU Inedex:', GPU)
    print(' frames num:', frames_num, ' learning rate:', learning_rate, )
    print('fold num:', frames_num, ' fold index:', fold_index)

    if not os.path.exists('./Result_log'):
        os.makedirs('./Result_log')

    log = Logger()
    log.open('./Result_log/' + Target_name + '_' + str(reTrain) + '_' + args.pt + '_intra_supervised' + str(
        args.test_percent) + '_log.txt', mode='a')
    log.write("\n----------------------------------------------- [START %s] %s\n\n" % (
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))

    # 运行媒介
    pynvml.nvmlInit()
    flag = 0
    max_g = []
    spaces = []
    GPU = '10'
    for gpu in range(8):
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_Gpu = meminfo.free / 1024 / 1024 / 1024
        if free_Gpu > 20:
            flag = 1
            GPU = str(gpu)
            print("GPU:", GPU)
            print("free_Gpu:", free_Gpu)
            max_g = GPU
            break
        print("GPU:", gpu)
        print("free_Gpu:", free_Gpu)

    # if free_Gpu < 40:
    # GPU = max_g.index(max(max_g))
    # batch_size = 10#int(150 / (47 / max_g[GPU] / 2))
    # GPU = str(GPU)
    if args.GPU != 10 and GPU == '10':
        GPU = str(args.GPU)
    if torch.cuda.is_available():
        device = torch.device('cuda:' + GPU if torch.cuda.is_available() else 'cpu')  #
        print('on GPU ', GPU)
    else:
        print('on CPU')

        # Target_db = MyDataset.Data_DG(root_dir=Target_saveRoot, dataName=Target_name, \
        #                               STMap=Target_map, frames_num=frames_num, args=args, domain_label=5,
        #                               datalist=datalist)
    Target_db = MyDataset.Data_Video(root_dir=Target_saveRoot, dataName=Target_name, \
                                     STMap=Target_map, frames_num=frames_num, args=args, domain_label=5)

    tgt_loader = DataLoader(Target_db, batch_size=batch_size_num, shuffle=False, num_workers=num_workers)

    my_model = SiNC()
    #my_model = RhythmFormer()

    # my_model.calculate_training_parameter_ratio()

    # my_model = ViT_ST_ST_Compact3_TDC_gra_sharp()

    if reTrain == 1:
        rPPGNet_name = '/remote-home/hao.lu/jywang/On-Road-rPPG/UnWeak-On-Road-rPPG-JY/pre_encoder/On-Road-rPPG_7000_unsupervised_SiNC'
        state_dict = torch.load(rPPGNet_name,
                                map_location=device)
        my_model = torch.load(
            rPPGNet_name,
            map_location=device)
        print('load ' + rPPGNet_name + ' right')
    my_model.to(device=device)

    # optimizer_rPPG = torch.optim.Adam(my_model.parameters(), lr=learning_rate)

    # loss_func_SP = MyLoss.SP_loss(device, clip_length=frames_num).to(device)

    tgt_iter = iter(tgt_loader)
    tgt_iter_per_epoch = len(tgt_iter)

    max_iter = args.max_iter
    start = timer()
    loss_res = {'bvp': [], 'hr': [], 'spo': [], 'rf': [], 'resp': [], 'IrrelevantPowerRatio': [],
                'NegativeMaxCrossCorr': [], 'NegPearsonsCorrLoss': [], 'NegSNRLoss': [],
                'IPR_SSL': [], 'EMD_SSL': [], 'SNR_SSL': []}
    loss_res_aug = {'bvp': [], 'hr': [], 'spo': [], 'rf': [], 'resp': [], 'IrrelevantPowerRatio': [],
                    'NegativeMaxCrossCorr': [], 'NegPearsonsCorrLoss': [], 'NegSNRLoss': [],
                    'IPR_SSL': [], 'EMD_SSL': [], 'SNR_SSL': []}
    eval_bvp_hr = {'MAE': [], 'RMSE': [], 'MER': [], 'P': []}
    eval_bvp_rr = {'MAE': [], 'RMSE': [], 'MER': [], 'P': []}
    eval_hr = {'MAE': [], 'RMSE': [], 'MER': [], 'P': []}
    eval_rr = {'MAE': [], 'RMSE': [], 'MER': [], 'P': []}
    eval_spo = {'MAE': [], 'RMSE': [], 'MER': [], 'P': []}
    eval_resp_rr = {'MAE': [], 'RMSE': [], 'MER': [], 'P': []}
    with tqdm(range(max_iter + 1)) as it:
        for iter_num in it:
            if iter_num == 0:
                # if iter_num % 500 == 0 and iter_num > 500:
                # temp = pd.DataFrame(loss_res)
                # temp.to_csv('loss_res_w_demintor.csv')
                # 测试
                print('Test:\n')

                # utils.loss_visual(loss_res,
                #                   './visuals/loss_visual/main_task_loss_key_' + args.pt + '_video_intra_supervised' + str(
                #                       args.test_percent) + '_' + args.tgt + '.png')
                loss_mean = []
                Label_pr = []
                Label_gt = []
                HR_pr_temp = []
                HR_rel_temp = []
                BVP_ALL = []
                BVP_PR_ALL = []
                Spo_pr_temp = []
                Spo_rel_temp = []
                RF_pr_temp = []
                RF_rel_temp = []
                Resp_ALL = []
                Resp_PR_ALL = []
                for step, (data, bvp, HR_rel, spo, rf, resp, _, _, _, _, _, _, _) in tqdm(enumerate(tgt_loader)):
                    data = Variable(data).float().to(device=device)
                    Wave = Variable(bvp).float().to(device=device)
                    HR_rel = Variable(HR_rel).float().to(device=device)
                    Spo_rel = Variable(spo).float().to(device=device)
                    RF_rel = Variable(rf).float().to(device=device)
                    Resp_rel = Variable(resp).float().to(device=device)
                    Wave = Wave.unsqueeze(dim=1)
                    Resp_rel = Resp_rel.unsqueeze(dim=1)
                    rand_idx = torch.randperm(data.shape[0])
                    # print(data.shape)
                    N, C, D, H, W = data.shape
                    # data = data.reshape(N * D, C, H, W)
                    Wave_pr = my_model(data)
                    Wave_pr = Wave_pr.reshape(N, 1, D)
                    # Wave_pr, HR_pr, Spo_pr, RF_pr, Resp_pre = my_model(data)
                    # print(Wave_pr.shape)

                    BVP_ALL.extend(Wave.data.cpu().numpy())
                    BVP_PR_ALL.extend(Wave_pr.data.cpu().numpy())
                    # HR_pr_temp.extend(HR_pr.data.cpu().numpy())
                    # HR_rel_temp.extend(HR_rel.data.cpu().numpy())

                # ME, STD, MAE, RMSE, MER, P = utils.MyEval(HR_pr_temp, HR_rel_temp)
                # print(Target_name)
                # log.write(
                #     'Test Inter HR:' \
                #     + ' | ME:  ' + str(ME) \
                #     + ' | STD: ' + str(STD) \
                #     + ' | MAE: ' + str(MAE) \
                #     + ' | RMSE: ' + str(RMSE) \
                #     + ' | MER: ' + str(MER) \
                #     + ' | P ' + str(P))
                # log.write('\n')
                # eval_hr['MAE'].append(MAE)
                # eval_hr['RMSE'].append(RMSE)
                # eval_hr['MER'].append(MER)
                # eval_hr['P'].append(P)

                ME, STD, MAE, RMSE, MER, P = utils.MyEval_bvp_hr(BVP_PR_ALL, BVP_ALL)
                log.write(
                    'Test Inter HR from BVP:' \
                    + ' | ME:  ' + str(ME) \
                    + ' | STD: ' + str(STD) \
                    + ' | MAE: ' + str(MAE) \
                    + ' | RMSE: ' + str(RMSE) \
                    + ' | MER: ' + str(MER) \
                    + ' | P ' + str(P))
                log.write('\n')
                eval_bvp_hr['MAE'].append(MAE)
                eval_bvp_hr['RMSE'].append(RMSE)
                eval_bvp_hr['MER'].append(MER)
                eval_bvp_hr['P'].append(P)
