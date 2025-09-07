import sys
print(sys.path)
import numpy as np
import pandas as pd
import scipy.io as scio

from base_model.GRU import MultiChannelGRU
from base_model.EEGNet import EEGNetModel
from base_model.REST_Model import LitNCA_EEG
from base_model.REST_Model_Conv import LitNCA_EEG2
import torch
from training_func.myTrainer import train as bs_train
from training_func.myTrainer import evaluate as bs_evaluate
from training_func.myTrainer import test as bs_test
from training_func.myTrainer import test_infer as bs_test_infer
from training_func.iEEG_trainer import EarlyStopping, train, evaluate, print_time_cost, test, test_infer

import joblib
from dataset import myDataset, iEEG_dataset

from scipy.fft import fft
import torch.nn as nn
import torch.optim as optim
import os 
import scipy.io as sio
import time
import warnings
import torch.nn.functional as F
from memory_profiler import profile
warnings.filterwarnings("ignore")
import sys
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader as gDataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import pickle
import gc
# 设置随机种子
torch.manual_seed(42)
pids = ['01','02','03','04','05', '06', '07', '08','09','10','11','12','13','14','15','16','17','18']
GPU_ID = 0
device = torch.device('cuda:{}'.format(GPU_ID) if torch.cuda.is_available() else "cpu")
print(device)
batch_size = 16

def padding_channel(data, channel=128, sample_rate = 512):
    if data.shape[1] > channel:
        # pad = np.zeros((data.shape[0], channel - data.shape[1], data.shape[2], data.shape[3]))
        # data = np.concatenate((data, pad), axis=1)
                # data = np.concatenate((data, pad), axis=1)
        data = data[:, :channel, :, :]
    
    while data.shape[3] > sample_rate:
        data = data[:, :, :, ::2]
    return data

def normal_data(data):
    # 计算 sample_point 维度上的最小值和最大值
    min_vals = np.min(data, axis=-1, keepdims=True)  # 形状: [numbers, channels, time_step, 1]
    max_vals = np.max(data, axis=-1, keepdims=True)  # 形状: [numbers, channels, time_step, 1]

    # 避免除零错误
    epsilon = 1e-8
    normalized_data = (data - min_vals) / (max_vals - min_vals + epsilon)
    return normalized_data


def train_func(pat_id, rate, device = device, step=2, dim=32,if_data=True ):

    infofile = '/net/inltitan1/scratch2/rpeng/long-term_dataset/ID'+pat_id+'/ID'+pat_id+'_info.mat'
    info = scio.loadmat(infofile)
    fs = info.get('fs')[0][0]
    T = 4

#stage one: cross-patient training
    non_seizure_train_list = []
    seizure_train_list = []
    non_seizure_val_list = []
    seizure_val_list = []
    ft_non_seizure_train_list = []
    ft_seizure_train_list = []
    ft_non_seizure_test_list = []
    ft_seizure_test_list = []
    for pid in pids:
        if pid is not pat_id:
            datafile = '/net/inltitan1/scratch2/rpeng/long-term_dataset/new_rate_'+str(rate)+'_T_'+str(T)+'s_Clips_of_pat_'+pid+'.pkl'
            with open(datafile, 'rb') as f:
                loaded_data = pickle.load(f)
                print(loaded_data.keys())
                no_seizure_clips_fea = loaded_data['non_seizure_clips']
                seizure_clips_fea = loaded_data['seizure_clips']

            no_seizure_clip_data = no_seizure_clips_fea['non_ictal_clips']
            # non_seizure_id = no_seizure_clips_fea['non_seizure_id']
            seizure_clip_data = seizure_clips_fea['ictal_clips']
            # seizure_id = seizure_clips_fea['seizure_id']

            print('len_seizure: ', len(seizure_clip_data ))
            print('len_no_seizure: ', len(no_seizure_clip_data ))

            l_seizure = len(seizure_clip_data)
            l_noseizure = len(no_seizure_clip_data)

            nonictal_data_train = padding_channel(np.concatenate(no_seizure_clip_data[:round(l_noseizure * 0.6)], axis=0))
            nonictal_data_val= padding_channel(np.concatenate(no_seizure_clip_data[round(l_noseizure * 0.6):], axis=0))
            ictal_data_train = padding_channel(np.concatenate(seizure_clip_data[:round(l_seizure * 0.6)], axis=0))
            ictal_data_val= padding_channel(np.concatenate(seizure_clip_data[round(l_seizure * 0.6):], axis=0))
            non_seizure_train_list.append(nonictal_data_train)
            non_seizure_val_list.append(nonictal_data_val)
            seizure_train_list.append(ictal_data_train)
            seizure_val_list.append(ictal_data_val)
        
        else:
            datafile = '/net/inltitan1/scratch2/rpeng/long-term_dataset/new_rate_'+str(rate)+'_T_'+str(T)+'s_Clips_of_pat_'+pid+'.pkl'
            with open(datafile, 'rb') as f:
                loaded_data = pickle.load(f)
                print(loaded_data.keys())
                no_seizure_clips_fea = loaded_data['non_seizure_clips']
                seizure_clips_fea = loaded_data['seizure_clips']

            ft_no_seizure_clip_data = no_seizure_clips_fea['non_ictal_clips']
            # ft_non_seizure_id = no_seizure_clips_fea['non_seizure_id']
            ft_seizure_clip_data = seizure_clips_fea['ictal_clips']
            # ft_seizure_id = seizure_clips_fea['seizure_id']

            print('len_seizure: ', len(seizure_clip_data ))
            print('len_no_seizure: ', len(no_seizure_clip_data ))

            ft_l_seizure = len(seizure_clip_data )
            ft_l_noseizure = len(no_seizure_clip_data )

            ft_nonictal_data_train = padding_channel(np.concatenate(ft_no_seizure_clip_data[:round(ft_l_noseizure * 0.3)], axis=0))
            ft_nonictal_data_test= padding_channel(np.concatenate(ft_no_seizure_clip_data[round(ft_l_noseizure * 0.3):], axis=0))
            ft_ictal_data_train = padding_channel(np.concatenate(ft_seizure_clip_data[:round(ft_l_seizure * 0.3)], axis=0))
            ft_ictal_data_test= padding_channel(np.concatenate(ft_seizure_clip_data[round(ft_l_seizure * 0.3):], axis=0))

            ft_non_seizure_train_list.append(ft_nonictal_data_train)
            ft_non_seizure_test_list.append(ft_nonictal_data_test)
            ft_seizure_train_list.append(ft_ictal_data_train)
            ft_seizure_test_list.append(ft_ictal_data_test)

    train_data_nonictal = np.concatenate(non_seizure_train_list, axis=0)
    train_data_ictal = np.concatenate(seizure_train_list, axis=0)
    val_data_nonictal = np.concatenate(non_seizure_val_list, axis=0)
    val_data_ictal = np.concatenate(seizure_val_list, axis=0)

    print(train_data_ictal.shape)
    print(train_data_nonictal.shape)

    channel, feature = train_data_ictal.shape[1], train_data_ictal.shape[3]
    num_ictal, num_ictal_v, num_non_ictal,num_non_ictal_v = train_data_ictal.shape[0], val_data_ictal.shape[0], train_data_nonictal.shape[0], val_data_nonictal.shape[0]

    resample_nonictal_id = torch.randperm(num_non_ictal)[:int(num_ictal * 1)]
    train_data = torch.Tensor(np.concatenate((train_data_ictal, train_data_nonictal[resample_nonictal_id]), axis=0))
    train_label = torch.Tensor(np.concatenate((np.ones(num_ictal), np.zeros(int(num_ictal * 1))), axis=0))

    resample_nonictal_id_v = torch.randperm(num_non_ictal_v)[:int(num_ictal_v * 1)]
    val_data = torch.Tensor(np.concatenate((val_data_ictal, val_data_nonictal[resample_nonictal_id_v]), axis=0))
    val_label = torch.Tensor(np.concatenate((np.ones(val_data_ictal.shape[0]), np.zeros(val_data_nonictal.shape[0])), axis=0))

    train_subset = myDataset.iEEG_Dataset(train_data, train_label, device=device)
    val_subset = myDataset.iEEG_Dataset(val_data, val_label, device=device)

    del train_data_ictal, val_data_ictal, val_data, val_label
    gc.collect()

    del train_data, train_label, train_data_nonictal, val_data_nonictal
    gc.collect()

    del no_seizure_clip_data, seizure_clip_data, no_seizure_clips_fea, seizure_clips_fea#, non_seizure_id,seizure_id
    gc.collect()

    sampler_t = WeightedRandomSampler(torch.tensor(train_subset.sample_weight), num_samples=len(train_subset.data), replacement=True)
    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=False, sampler=sampler_t, drop_last=True, num_workers=6)
    val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=6)

    del train_subset, val_subset
    gc.collect()

    # model = MultiChannelGRU(input_size = int(feature), hidden_size=32, num_layers=2, output_size=1, T=T, channels= channel,device=device).to(device)
    model = EEGNetModel(chans=int(channel), classes=1, time_points=int(T*feature)).to(device)
    # 将类别权重传入损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-5, betas=(0.9, 0.999))
    # criterion =  nn.MSELoss()
    criterion =  nn.BCEWithLogitsLoss()
    save_path = '/home/rpeng/RPeng_workspace/TEST_0918/REST_per_patient/cross_patient_pretraining/save_models_'+str(if_data)+'_data/rate_'+str(rate)+'/EEGNet/'+'T_'+str(T)+'/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    early_stopping = EarlyStopping(patience=10, verbose=True, path=os.path.join(save_path, 'pat_'+str(pat_id)+'_EEGNet_checkpoint_nonorm.pt'))
    start_epoch = 0
    train_ls, val_ls = [], []

    for epoch in range(start_epoch, 200):
        since = time.time()
        train_loss_A= bs_train(model=model, optimizer=optimizer, dataloader=train_dataloader, criterion = criterion, device=device, modeltype='EEGNet')
        val_loss_A= bs_evaluate(model=model, dataloader=val_dataloader, criterion = criterion, device=device, modeltype='EEGNet')
        
        train_ls.append(train_loss_A)
        val_ls.append(val_loss_A)
        print('#Model: epoch:%02d train_loss:%.3e val_loss:%0.3e time:%s'% (epoch, train_loss_A, val_loss_A,  print_time_cost(since)))

        early_stopping(val_loss_A, model)
        if early_stopping.early_stop:
            print("Early stopping")
            best_epoch = epoch
            print('Best epoch is: ', best_epoch)
            break
    print("Model: final epoch: train loss %f, test loss %f\n" % (train_ls[-1], val_ls[-1]))
    epochs_list = range(1, len(train_ls)+1)
    # 绘制训练损失和验证损失
    plt.figure(figsize=(10, 6))
    # 绘制训练损失曲线
    plt.plot(epochs_list, train_ls, label='Training Loss', color='blue', linestyle='-', marker='o')
    # 绘制验证损失曲线
    plt.plot(epochs_list, val_ls, label='Validation Loss', color='red', linestyle='-', marker='x')
    # 添加标题和标签
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    # 显示图例
    plt.legend()
    fig_name = save_path+'Data_'+str(if_data)+'_Rate_'+str(rate)+'_pid_'+str(pat_id)+'T_'+str(T)+'_EEGNet_'+'_Loss_pretraining.jpg'
    plt.savefig(fig_name)
    plt.plot()

    del train_dataloader, val_dataloader
    gc.collect()

#stage two: finetuning on patient

    ft_train_data_nonictal = np.concatenate(ft_non_seizure_train_list, axis=0)
    ft_train_data_ictal = np.concatenate(ft_seizure_train_list, axis=0)
    ft_test_data_nonictal = np.concatenate(ft_non_seizure_test_list, axis=0)
    ft_test_data_ictal = np.concatenate(ft_seizure_test_list, axis=0)

    print(ft_train_data_ictal.shape)
    print(ft_train_data_nonictal.shape)

    ft_channel, ft_feature = ft_train_data_ictal.shape[1], ft_train_data_ictal.shape[3]
    ft_num_ictal, ft_num_non_ictal = ft_train_data_ictal.shape[0], ft_train_data_nonictal.shape[0]
    ft_resample_nonictal_id = torch.randperm(ft_num_non_ictal)[:int(ft_num_ictal * 1)]
    ft_train_data = torch.Tensor(np.concatenate((ft_train_data_ictal, ft_train_data_nonictal[ft_resample_nonictal_id]), axis=0))
    ft_train_label = torch.Tensor(np.concatenate((np.ones(ft_num_ictal), np.zeros(int(ft_num_ictal * 1))), axis=0))

    ft_test_data = torch.Tensor(np.concatenate((ft_test_data_ictal, ft_test_data_nonictal), axis=0))
    ft_test_label = torch.Tensor(np.concatenate((np.ones(ft_test_data_ictal.shape[0]), np.zeros(ft_test_data_nonictal.shape[0])), axis=0))

    ft_train_subset = myDataset.iEEG_Dataset(ft_train_data, ft_train_label, device=device)

    del ft_train_data_ictal, ft_test_data_ictal
    gc.collect()

    del ft_train_data, ft_train_label, ft_train_data_nonictal, ft_test_data_nonictal
    gc.collect()

    del ft_no_seizure_clip_data, ft_seizure_clip_data, ft_non_seizure_id,ft_seizure_id
    gc.collect()

    ft_sampler_t = WeightedRandomSampler(torch.tensor(ft_train_subset.sample_weight), num_samples=len(ft_train_subset.data), replacement=True)
    ft_train_dataloader = DataLoader(ft_train_subset, batch_size=batch_size, shuffle=False, sampler=ft_sampler_t, drop_last=True, num_workers=6)

    del ft_train_subset
    gc.collect()

    print('Pre-trained model loading...')
    # ft_model = MultiChannelGRU(input_size = int(ft_feature), hidden_size=32, num_layers=2, output_size=1, T=T, channels= ft_channel,device=device).to(device)
    ft_model = EEGNetModel(chans=int(ft_channel), classes=1, time_points=int(T*ft_feature)).to(device)

    ft_model.load_state_dict(torch.load(os.path.join(save_path, 'pat_'+str(pat_id)+'_EEGNet_checkpoint_nonorm.pt')))
    ft_model.to(device).train()
    print('Pre-trained model loaded...')
    ft_optimizer = torch.optim.Adam(ft_model.parameters(), lr=5e-5, betas=(0.9, 0.999))
    # criterion =  nn.MSELoss()
    ft_criterion =  nn.BCEWithLogitsLoss()
    ft_save_path = '/home/rpeng/RPeng_workspace/TEST_0918/REST_per_patient/cross_patient_finetuning/save_models_'+str(if_data)+'_data/rate_'+str(rate)+'/EEGNet/'+'T_'+str(T)+'/'

    if not os.path.exists(ft_save_path):
        os.makedirs(ft_save_path)
    start_epoch = 0
    ft_train_ls= []

    print('Finetuning Training...')
    best_loss = 10000
    for epoch in range(start_epoch, 200):
        since = time.time()
        train_loss_A= bs_train(model=ft_model, optimizer=ft_optimizer, dataloader=ft_train_dataloader, criterion = ft_criterion, device=device, modeltype='EEGNet')
        ft_train_ls.append(train_loss_A)
        print('#Model: epoch:%02d train_loss:%.3e time:%s'% (epoch, train_loss_A,  print_time_cost(since)))
        if best_loss > train_loss_A:
            best_loss = train_loss_A
            torch.save(ft_model.state_dict(), os.path.join(ft_save_path, 'pat_'+str(pat_id)+'_EEGNet_checkpoint_nonorm.pt'))
            print('Model saved...')
    print("Model: final epoch: train loss %f\n" % (ft_train_ls[-1]))
    epochs_list = range(1, len(ft_train_ls)+1)
    # 绘制训练损失和验证损失
    plt.figure(figsize=(10, 6))
    # 绘制训练损失曲线
    plt.plot(epochs_list, ft_train_ls, label='Training Loss', color='blue', linestyle='-', marker='o')

    # 添加标题和标签
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # 显示图例
    plt.legend()
    fig_name = ft_save_path+'Data_'+str(if_data)+'_Rate_'+str(rate)+'_pid_'+str(pat_id)+'T_'+str(T)+'_EEGNet_'+'_Loss_finetuning.jpg'
    plt.savefig(fig_name)
    plt.plot()

    del ft_train_dataloader
    gc.collect()
    print('Model finetuned...')

#test   
    print('Testing...')
    print('preparing data...')

    print(ft_test_data.shape)
    print(ft_test_label.shape)
    print('data prepared...')

    test_subset = myDataset.iEEG_Dataset(ft_test_data, ft_test_label,device=device)
    test_dataloader = DataLoader(test_subset, batch_size=16, shuffle=False, num_workers=2)
    test_class_weight = test_subset.class_weight
    test_sample_weight = test_subset.sample_weight 
    
    print('Pretrained model testing...')
    teacher_test_precision, teacher_test_recall, teacher_test_f1, teacher_test_bca, teacher_test_spec,teacher_test_fpr = bs_test(model, test_dataloader, device=device, modeltype='EEGNet', sample_weight = test_sample_weight)
    teacher_inference_time_sample = bs_test_infer(model, test_dataloader, device=device, modeltype='EEGNet')
    teacher_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Pretrained Model #parameter: ', teacher_total_params)
    print('Pretrained Model current precision:', teacher_test_precision,' recall:',  teacher_test_recall, ' f1:', teacher_test_f1)
    print('Pretrained Model current BCA:',teacher_test_bca,' specificity:',  teacher_test_spec, ' fpr:', teacher_test_fpr)
    print('Pretrained Model inference time per sample:', teacher_inference_time_sample)
    print('Pretrained Model tested')

    print('Finetuned model testing...')
    tuned_test_precision, tuned_test_recall, tuned_test_f1, tuned_test_bca, tuned_test_spec,tuned_test_fpr = bs_test(ft_model, test_dataloader, device=device, modeltype='EEGNet', sample_weight = test_sample_weight)
    tuned_inference_time_sample = bs_test_infer(ft_model, test_dataloader, device=device, modeltype='EEGNet')
    tuned_total_params = sum(p.numel() for p in ft_model.parameters() if p.requires_grad)
    
    total_params = sum(p.numel() for p in ft_model.parameters() if p.requires_grad)
    print('Tuned Model #parameter: ', tuned_total_params)
    print('Tuned Model current precision:', tuned_test_precision,' recall:', tuned_test_recall, ' f1:', tuned_test_f1)
    print('Tuned Model current BCA:',tuned_test_bca,' specificity:', tuned_test_spec, ' fpr:', tuned_test_fpr)
    print('Tuned Model inference time per sample:', tuned_inference_time_sample)
    print('Tuned model tested')

    return [pat_id, rate, T, step, dim, teacher_test_precision, teacher_test_recall, teacher_test_f1, teacher_test_bca, teacher_test_spec,teacher_test_fpr, teacher_inference_time_sample, teacher_total_params], \
            [pat_id, rate, T, step, dim, tuned_test_precision, tuned_test_recall, tuned_test_f1, tuned_test_bca, tuned_test_spec,tuned_test_fpr, tuned_inference_time_sample, tuned_total_params]


def exclewriter(pid, rate_ls, if_data, dim_ls):
    results_p, results_tuned = [],[]
    for rate in rate_ls:
        for dim in dim_ls:
            pre_train, finetune = train_func(pid, rate, dim=dim, if_data = if_data)
            results_p.append(pre_train)
            results_tuned.append(finetune)
    return results_p, results_tuned

def execute_tune(pid, rate_ls, dim_ls, if_data=True):
    excel_file = 'Cross_patient_training_finetuning_Pat_'+pid+"_model_EEGNet_Data_"+str(if_data)+"_results.xlsx"
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:

        data_pre, data_tuned = exclewriter(pid, rate_ls, if_data, dim_ls)
        # 转换为 Pandas DataFrame
        df_pre = pd.DataFrame(data_pre, columns=["Patient", "RATE", "T", 'step',"Dim","Precision", "Recall", "F1 Score", "BCA","Specificity","FPR",'inference_time_sample','total_params'])
        df_tuned = pd.DataFrame(data_tuned, columns=["Patient", "RATE", "T", 'step',"Dim","Precision", "Recall", "F1 Score", "BCA","Specificity","FPR",'inference_time_sample','total_params'])

        # 写入每个模型的结果到单独的 Sheet
        df_pre.to_excel(writer, sheet_name='EEGNet_pre', index=False)
        df_tuned.to_excel(writer, sheet_name='EEGNet_tuned', index=False)
    print(f"Results saved to {excel_file}")