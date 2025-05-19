import pandas as pd
import scipy.io as io
import torch
import torch.nn as nn
import numpy as np
import MyDataset
import MyLoss
import Model
import Baseline
import PhysMLE
from Intra_Model import BVPNet
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
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
import warnings

# from PCGrad.pcgrad import PCGrad

warnings.simplefilter('ignore')

TARGET_DOMAIN = {'VIPL': ['VIPL'], \
                 'V4V': ['V4V'], \
                 'PURE': ['PURE'], \
                 'BUAA': ['BUAA'], \
                 'UBFC': ['UBFC'], \
                 'HCW': ['HCW'],
                 'VV100': ['VV100'],
                 'MMPD': ['MMPD'],
                 'HMPC-Dv1': ['HMPC-Dv1'],
                 'On_Road_rPPG': ['On-Road-rPPG-Processed-Final']}

FILEA_NAME = {'VIPL': ['VIPL', 'VIPL', 'STMap_RGB_Align_CSI'], \
              'V4V': ['V4V', 'V4V', 'STMap_RGB'], \
              'PURE': ['PURE', 'PURE', 'STMap'], \
              'BUAA': ['BUAA', 'BUAA', 'STMap_RGB'], \
              'UBFC': ['UBFC', 'UBFC', 'STMap'], \
              'HCW': ['HCW', 'HCW', 'STMap_RGB'],
              'VV100': ['VV100', 'VV100', 'STMap_RGB'],
              'MMPD': ['MMPD', 'MMPD', 'STMap_RGB'],
              'HMPC-Dv1': ['HMPC-Dv1', 'HMPC-Dv1', 'STMap'],
              'On_Road_rPPG': ['On-Road-rPPG', 'On-Road-rPPG',
                               'STMap_RGB']
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
    root_file = r'/home/jywang/Data/'
    frames_num = args.frames_num
    # 参数

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
                                           Target_saveRoot, Target_map, 10, frames_num)

    train_list, test_list = MyDataset.CrossValidation(Target_saveRoot, fold_num=args.fold_num,
                                                      test_percent=args.test_percent)

    # 训练参数
    batch_size_num = args.batchsize
    epoch_num = args.epochs
    learning_rate = args.lr

    test_batch_size = args.batchsize
    num_workers = args.num_workers
    GPU = args.GPU

    # 图片参数
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
    log.open('./Result_log/' + Target_name + '_' + str(reTrain) + '_' + args.pt + 'cross_BUAA_fulltest' + str(
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

    source_db_0 = MyDataset.Data_DG(root_dir=Target_saveRoot, dataName=Target_name, \
                                    STMap=Target_map, frames_num=frames_num, args=args, domain_label=0)
    Target_db = MyDataset.Data_DG(root_dir=Target_saveRoot, dataName=Target_name, \
                                  STMap=Target_map, frames_num=frames_num, args=args, domain_label=5,
                                    datalist=test_list)

    src_loader_0 = DataLoader(source_db_0, batch_size=batch_size_num, shuffle=True, num_workers=num_workers)
    tgt_loader = DataLoader(Target_db, batch_size=batch_size_num, shuffle=False, num_workers=num_workers)

    # my_model = Baseline.BaseNet_CNN()
    my_model = BVPNet.BVPNet()
    # my_model = RhythmNet.RhythmNet()
    # my_model = PhysMLE.BaseNet_CNN()
    # my_model.calculate_training_parameter_ratio()

    if reTrain == 1:
        my_model = torch.load('/home/jywang/project/STMap_Baseline_On_Road_rPPG/Result_Model/BUAA_10000_resnet18_full_supervised',
                              map_location=device)
        print('load ' + rPPGNet_name + ' right')

    my_model.to(device=device)

    # input = torch.randn(1, 3, 224, 224)
    # input = input.float().to(device=device)
    # flops, params = profile(my_model, inputs=(input,))
    # print("计算量flops: %.2fG" % (flops / (1024 ** 3)))

    optimizer_rPPG = torch.optim.Adam(my_model.parameters(), lr=learning_rate)
    # optimizer_rPPG = torch.optim.Adam(rppg_params + other_params, lr=learning_rate)
    # optimizer_spo = torch.optim.Adam(spo_params + other_params, lr=learning_rate)

    loss_func_BVP = [MyLoss.P_loss3().to(device), nn.SmoothL1Loss().to(device)]
    loss_func_RESP = [MyLoss.P_loss3().to(device), nn.SmoothL1Loss().to(device)]
    loss_func_SPO = nn.SmoothL1Loss().to(device)
    loss_func_RF = nn.SmoothL1Loss().to(device)
    loss_func_L1 = nn.SmoothL1Loss().to(device)

    # loss_func_compare = nn.MSELoss(reduction='sum').to(device)
    loss_func_SP = MyLoss.SP_loss(device, clip_length=frames_num).to(device)

    # loss_func_SP = MyLoss.SP_loss(device, clip_length=frames_num).to(device)
    src_iter_0 = src_loader_0.__iter__()
    src_iter_per_epoch_0 = len(src_iter_0)

    tgt_iter = iter(tgt_loader)
    tgt_iter_per_epoch = len(tgt_iter)

    max_iter = args.max_iter
    start = timer()
    loss_res = {'bvp': [], 'hr': [], 'spo': [], 'rf': [], 'resp': [], 'all': [],
                }

    eval_bvp_hr = {'MAE': [], 'RMSE': [], 'MER': [], 'P': []}
    eval_hr = {'MAE': [], 'RMSE': [], 'MER': [], 'P': []}
    eval_rr = {'MAE': [], 'RMSE': [], 'MER': [], 'P': []}
    eval_spo = {'MAE': [], 'RMSE': [], 'MER': [], 'P': []}
    eval_resp_rr = {'MAE': [], 'RMSE': [], 'MER': [], 'P': []}

    with tqdm(range(max_iter + 1)) as it:
        for iter_num in it:
            my_model.train()
            if (iter_num % src_iter_per_epoch_0 == 0):
                src_iter_0 = src_loader_0.__iter__()

            ######### data prepare #########
            data0, bvp0, HR_rel0, spo_rel0, resp_rel0, rf_rel0, data_aug0, bvp_aug0, HR_rel_aug0, spo_rel_aug0, resp_rel_aug0, rf_rel_aug0, domain_label0 = src_iter_0.__next__()

            data0 = Variable(data0).float().to(device=device)
            bvp0 = Variable(bvp0).float().to(device=device).unsqueeze(dim=1)
            HR_rel0 = Variable(torch.Tensor(HR_rel0)).float().to(device=device)
            spo_rel0 = Variable(torch.Tensor(spo_rel0)).float().to(device=device)
            rf_rel0 = Variable(torch.Tensor(rf_rel0)).float().to(device=device)
            resp_rel0 = Variable(resp_rel0).float().to(device=device).unsqueeze(dim=1)

            data_aug0 = Variable(data_aug0).float().to(device=device)
            bvp_aug0 = Variable(bvp_aug0).float().to(device=device).unsqueeze(dim=1)
            HR_rel_aug0 = Variable(torch.Tensor(HR_rel_aug0)).float().to(device=device)
            spo_rel_aug0 = Variable(torch.Tensor(spo_rel_aug0)).float().to(device=device)
            rf_rel_aug0 = Variable(torch.Tensor(rf_rel_aug0)).float().to(device=device)
            resp_rel_aug0 = Variable(resp_rel_aug0).float().to(device=device).unsqueeze(dim=1)

            optimizer_rPPG.zero_grad()

            input = data0
            input_aug = data_aug0

            bvp_pre_zip, HR_pr_zip, spo_pre_zip, resp_pre_zip, rf_pre_zip = my_model(input)

            bvp_pre_aug_zip, HR_pr_aug_zip, spo_pre_aug_zip, resp_pre_aug_zip, rf_pre_aug_zip = my_model(input_aug)

            rppg_loss_0, loss_res = MyLoss.get_loss(bvp_pre_zip, resp_pre_zip, HR_pr_zip, rf_pre_zip, spo_pre_zip, \
                                                    bvp0, resp_rel0, HR_rel0, rf_rel0, spo_rel0, Target_name,
                                                    loss_func_BVP,
                                                    loss_func_RESP, loss_func_L1,
                                                    loss_func_RF, loss_func_SPO, args, iter_num, loss_res
                                                    )

            rppg_loss_aug_0, _ = MyLoss.get_loss(bvp_pre_aug_zip, resp_pre_aug_zip, HR_pr_aug_zip, rf_pre_aug_zip,
                                                 spo_pre_aug_zip, \
                                                 bvp_aug0, resp_rel_aug0, HR_rel_aug0, rf_rel_aug0, spo_rel_aug0,
                                                 Target_name, loss_func_BVP, loss_func_RESP, loss_func_L1,
                                                 loss_func_RF, loss_func_SPO, args, iter_num
                                                 )

            rppg_loss = rppg_loss_0 + rppg_loss_aug_0

            # k = 2.0 / (1.0 + np.exp(-10.0 * iter_num / args.max_iter)) - 1.0
            loss_all = rppg_loss  # + 0.1 * k * loss_TA + 0.001 * k * loss_CM + 0.01 * k * loss_DM

            if torch.sum(torch.isnan(rppg_loss)) > 0:
                print('Nan')
                continue
            else:
                rppg_loss.backward()
                optimizer_rPPG.step()
                # spo_loss.backward()
                # optimizer_spo.step()
                it.set_postfix(
                    ordered_dict={
                        "Train Inter": iter_num,
                        "rppg loss": rppg_loss.data.cpu().numpy(),
                        # "spo loss": spo_loss.data.cpu().numpy()
                    },
                    refresh=False,
                )

            log.write(
                'Train Inter:' + str(iter_num) \
                + ' | Overall Loss:  ' + str(loss_all.data.cpu().numpy()) \
                + ' | rppg Loss:  ' + str(rppg_loss.data.cpu().numpy()) \
                # + ' | TA Loss:  ' + str(loss_TA.data.cpu().numpy()) \
                + ' |' + time_to_str(timer() - start, 'min'))
            log.write('\n')

            if (iter_num >0) and (iter_num % 1000 == 0):
                # if iter_num % 500 == 0 and iter_num > 500:
                # temp = pd.DataFrame(loss_res)
                # temp.to_csv('loss_res_w_demintor.csv')
                # 测试
                print('Test:\n')

                # utils.loss_visual(loss_res,
                #                   './visuals/loss_visual/main_task_loss_key_' + args.pt + '_' + args.tgt + '_' + str(
                #                       args.lr) + 'cross_BUAA_fulltest.png')

                my_model.eval()
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
                for step, (data, bvp, HR_rel, spo, resp, rf, _, _, _, _, _, _, _) in tqdm(enumerate(tgt_loader)):
                    data = Variable(data).float().to(device=device)
                    Wave = Variable(bvp).float().to(device=device)
                    Resp = Variable(resp).float().to(device=device)
                    HR_rel = Variable(HR_rel).float().to(device=device)
                    Spo_rel = Variable(spo).float().to(device=device)
                    RF_rel = Variable(rf).float().to(device=device)
                    Wave = Wave.unsqueeze(dim=1)
                    Resp = Resp.unsqueeze(dim=1)
                    rand_idx = torch.randperm(data.shape[0])
                    Wave_pr, HR_pr, Spo_pr, RESP_pr, RF_pr = my_model(data)

                    HR_rel_temp.extend(HR_rel.data.cpu().numpy())
                    # temp, HR_pr = loss_func_SP(Wave_pr, HR_pr)
                    HR_pr_temp.extend(HR_pr.data.cpu().numpy())
                    RF_pr_temp.extend(RF_pr.data.cpu().numpy())
                    RF_rel_temp.extend(RF_rel.data.cpu().numpy())
                    BVP_ALL.extend(Wave.data.cpu().numpy())
                    BVP_PR_ALL.extend(Wave_pr.data.cpu().numpy())
                    # Resp_ALL.extend(Resp.data.cpu().numpy())
                    # Resp_PR_ALL.extend(RESP_pr.data.cpu().numpy())
                    Spo_pr_temp.extend(Spo_pr.data.cpu().numpy())
                    Spo_rel_temp.extend(Spo_rel.data.cpu().numpy())

                # print('HR:')
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

                ME, STD, MAE, RMSE, MER, P = utils.MyEval_bvp_hr(BVP_PR_ALL, BVP_ALL)
                log.write(
                    'Test Inter HR from BVP:' + str(iter_num) \
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

                ME, STD, MAE, RMSE, MER, P = utils.MyEval(Spo_pr_temp, Spo_rel_temp)
                log.write(
                    'Test Inter SPO2:' + str(iter_num) \
                    + ' | ME:  ' + str(ME) \
                    + ' | STD: ' + str(STD) \
                    + ' | MAE: ' + str(MAE) \
                    + ' | RMSE: ' + str(RMSE) \
                    + ' | MER: ' + str(MER) \
                    + ' | P ' + str(P))
                log.write('\n')
                eval_spo['MAE'].append(MAE)
                eval_spo['RMSE'].append(RMSE)
                eval_spo['MER'].append(MER)
                eval_spo['P'].append(P)

                # ME, STD, MAE, RMSE, MER, P = utils.MyEval_resp_rr(Resp_PR_ALL, Resp_ALL)
                # log.write(
                #     'Test Inter RR from RESP:' + str(iter_num) \
                #     + ' | ME:  ' + str(ME) \
                #     + ' | STD: ' + str(STD) \
                #     + ' | MAE: ' + str(MAE) \
                #     + ' | RMSE: ' + str(RMSE) \
                #     + ' | MER: ' + str(MER) \
                #     + ' | P ' + str(P))
                # log.write('\n')
                # eval_resp_rr['MAE'].append(MAE)
                # eval_resp_rr['RMSE'].append(RMSE)
                # eval_resp_rr['MER'].append(MER)
                # eval_resp_rr['P'].append(P)

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

                if not os.path.exists('./visuals/intra_visual/'):
                    os.makedirs('./visuals/intra_visual/')
                eval_hr_save = pd.DataFrame(eval_hr)
                eval_hr_save.to_csv('./visuals/intra_visual/' + Target_name + '_' + str(
                    reTrain) + 'BVPNet_fulltest' + '_HR.csv')
                eval_bvp_hr_save = pd.DataFrame(eval_bvp_hr)
                eval_bvp_hr_save.to_csv('./visuals/intra_visual/' + Target_name + '_' + str(
                    reTrain) + '_' + args.pt + 'BVPNet_fulltest' + '_BVP_HR.csv')
                # eval_resp_rr_save = pd.DataFrame(eval_resp_rr)
                # eval_resp_rr_save.to_csv('./visuals/intra_visual/' + Target_name + '_' + str(
                #     reTrain) + '_' + args.pt + 'BVPNet_fulltest' + '_RESP_RR.csv')
                eval_spo_save = pd.DataFrame(eval_spo)
                eval_spo_save.to_csv('./visuals/intra_visual/' + Target_name + '_' + str(
                    reTrain) + 'BVPNet_fulltest' + '_SPO.csv')
                eval_rr_save = pd.DataFrame(eval_rr)
                eval_rr_save.to_csv('./visuals/intra_visual/' + Target_name + '_' + str(
                    reTrain) + 'BVPNet_fulltest' + '_RR.csv')

                # if not os.path.exists('./Result_Model'):
                #     os.makedirs('./Result_Model')
                # #
                # # if not os.path.exists('./Result'):
                # #     os.makedirs('./Result')
                # # io.savemat('./Result/' + Target_name + '_' + str(iter_num) + '_' + str(
                # #     reTrain) + '_' + args.pt + '_intra_supervised' + str(args.test_percent) + '_HR_pr.mat',
                # #            {'HR_pr': HR_pr_temp})
                # # io.savemat('./Result/' + Target_name + '_' + str(iter_num) + '_' + str(
                # #     reTrain) + '_' + args.pt + '_intra_supervised' + str(args.test_percent) + '_HR_rel.mat',
                # #            {'HR_rel': HR_rel_temp})
                # # if Target_name in ['VIPL', 'PURE', 'VV100', 'HMPC-Dv1']:
                # #     io.savemat('./Result/' + Target_name + '_' + str(iter_num) + '_' + str(
                # #         reTrain) + '_' + args.pt + '_intra_supervised' + str(args.test_percent) + '_SPO_pr.mat',
                # #                {'SPO_pr': Spo_pr_temp})
                # #     io.savemat('./Result/' + Target_name + '_' + str(iter_num) + '_' + str(
                # #         reTrain) + '_' + args.pt + '_intra_supervised' + str(args.test_percent) + '_SPO_rel.mat',
                # #                {'SPO_rel': Spo_rel_temp})
                # # elif Target_name in ['HCW', 'V4V']:
                # #     io.savemat('./Result/' + Target_name + '_' + str(iter_num) + str(iter_num) + '_' + str(
                # #         reTrain) + '_' + args.pt + '_intra_supervised' + str(args.test_percent) + '_RF_pr.mat',
                # #                {'RF_pr': RF_pr_temp})
                # #     io.savemat('./Result/' + Target_name + '_' + str(iter_num) + '_' + str(
                # #         reTrain) + '_' + args.pt + '_intra_supervised' + str(args.test_percent) + '_RF_rel.mat',
                # #                {'RF_rel': RF_rel_temp})
                # #
                # # if Target_name not in ['VIPL', 'V4V', 'HMPC-Dv1']:
                # #     io.savemat('./Result/' + Target_name + '_' + str(iter_num) + '_' + str(
                # #         reTrain) + '_' + args.pt + '_intra_supervised' + str(args.test_percent) + '_WAVE_ALL.mat',
                # #                {'Wave': BVP_ALL})
                # #     io.savemat('./Result/' + Target_name + '_' + str(iter_num) + '_' + str(
                # #         reTrain) + '_' + args.pt + '_intra_supervised' + str(args.test_percent) + '_WAVE_PR_ALL.mat',
                # #                {'Wave': BVP_PR_ALL})
                torch.save(my_model,
                           './Result_Model/' + Target_name + '_' + str(iter_num)  + '_full_supervised_BVPNet')
                print('saveModel As ' + Target_name + '_' + str(iter_num) + '_full_supervised_BVPNet')
