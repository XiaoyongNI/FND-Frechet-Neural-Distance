import sys
print(sys.path)
import numpy as np
import pandas as pd
import scipy.io as scio

from models.EEGNet import EEGNetModel
from models.Chrononet import ChronoNet

import torch
from training_func.myTrainer import train as bs_train
from training_func.myTrainer import evaluate as bs_evaluate
from training_func.myTrainer import test as bs_test
from training_func.myTrainer import test_infer as bs_test_infer
from training_func.iEEG_trainer import EarlyStopping, train, evaluate, print_time_cost, test, test_infer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

import joblib
from dataset import myDataset

from scipy.fft import fft
import torch.nn as nn
import torch.optim as optim
import os 
import scipy.io as sio
import time
import warnings
import torch.nn.functional as F
warnings.filterwarnings("ignore")
import sys
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader as gDataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import pickle
import gc

torch.manual_seed(42)
pids = ['01','02','03','04','05', '06', '07', '08','09','10','11','12','13','14','15','16','17','18']
GPU_ID = 0
device = torch.device('cuda:{}'.format(GPU_ID) if torch.cuda.is_available() else "cpu")
print(device)
batch_size = 16


def train_func(pat_id, rate, device = device, step=2, dim=32,if_data=True, if_artifact=True):
    infofile = '/net/inltitan1/scratch2/rpeng/long-term_dataset/ID'+pat_id+'/ID'+pat_id+'_info.mat'
    info = scio.loadmat(infofile)
    fs = info.get('fs')[0][0]
    T = 1
    if if_data:#new_rate_0_T_4s_Clips_of_pat_02
        datafile = '/net/inltitan1/scratch2/rpeng/long-term_dataset/stride01_new_rate_'+str(rate)+'_T_'+str(T)+'s_Clips_of_pat_'+pat_id+'.pkl'
    else:
        datafile = '/net/inltitan1/scratch2/rpeng/long-term_dataset/stride01_new_rate_'+str(rate)+'_T_'+str(T)+'s_Feas_of_pat_'+pat_id+'.pkl'
    with open(datafile, 'rb') as f:
        loaded_data = pickle.load(f)
        print(loaded_data.keys())
        data = loaded_data['data'] if if_data else loaded_data['features']
        labels = loaded_data['labels']
    
    print('data shape: ', np.shape(data))#(454, 88, 4, 512)
    print('labels shape: ', np.shape(labels))#(454,)

    print('len_seizure: ', len(data ))#454
    print('len_no_seizure: ', len(labels ))#454

    data = np.stack(data, axis=0)
    labels = np.stack(labels, axis=0)

    # data = data.reshape(-1, 1, 1, 512)  # (454 * 88, 4, 512)
    # labels = np.repeat(labels, 88)  # (454 * 88,)


    print('data shape: ', np.shape(data))#(454, 88, 4, 512)
    print('labels shape: ', np.shape(labels))#(454,)


    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    results = []

    train_idx, test_idx = next(skf.split(X = data, y=labels))
    X_train, X_test = data[train_idx], data[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    train_data, val_data, train_label, val_label = train_test_split(X_train, y_train,  test_size=0.2, stratify=y_train, random_state=42)
    train_data, val_data, test_data = torch.tensor(train_data), torch.tensor(val_data), torch.tensor(X_test)
    train_label, val_label, test_label = torch.tensor(train_label), torch.tensor(val_label),torch.tensor(y_test)
    print('train_data shape: ', train_data.shape)
    print('train_label shape: ', train_label.shape)
    channel = train_data.shape[1]
    feature = train_data.shape[3]
    print('channel: ', channel)
    print('feature: ', feature)

    train_subset = myDataset.iEEG_Dataset(train_data, train_label, device=device)
    val_subset = myDataset.iEEG_Dataset(val_data, val_label, device=device)
    test_subset = myDataset.iEEG_Dataset(test_data, test_label, device=device)
    sampler_t = WeightedRandomSampler(torch.tensor(train_subset.sample_weight), num_samples=len(train_subset.data), replacement=True)

    train_dataloader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler_t, num_workers=6, drop_last=True)
    val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)

    print('train_dataloader shape: ', train_dataloader.dataset.data.shape)
    print('val_dataloader shape: ', val_dataloader.dataset.data.shape)

    model = ChronoNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-3, betas=(0.9, 0.999))
    criterion =  nn.BCEWithLogitsLoss()
    save_path = 'BrainBERT/checkpoints/save_models_'+str(if_data)+'_data/rate_'+str(rate)+'/EEGNet/'+'T_'+str(T)+'/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    early_stopping = EarlyStopping(patience=10, verbose=True, path=os.path.join(save_path, 'pat_'+str(pat_id)+'_ChronoNet_checkpoint.pt'))
    start_epoch = 0
    train_ls, val_ls = [], []
    for epoch in range(start_epoch, 50):
        since = time.time()
        train_loss_A= bs_train(model=model, optimizer=optimizer, dataloader=train_dataloader, criterion = criterion, device=device, modeltype='EEGNet')
        val_loss_A,_= bs_evaluate(model=model, dataloader=val_dataloader, criterion = criterion, device=device, modeltype='EEGNet')
        train_ls.append(train_loss_A)
        val_ls.append(val_loss_A)
        print('#Model: epoch:%02d train_loss:%.3e val_loss:%.3e time:%s'% (epoch, train_loss_A, val_loss_A,  print_time_cost(since)))
        early_stopping(val_loss_A, model)
        if early_stopping.early_stop:
            print("Early stopping")
            best_epoch = epoch
            print('Best epoch is: ', best_epoch)
            break
    print("Model: final epoch: train loss %f val loss %f\n" % (train_ls[-1], val_ls[-1]))
    epochs_list = range(1, len(train_ls)+1)
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_list, train_ls, label='Training Loss', color='blue', linestyle='-', marker='o')
    plt.plot(epochs_list, val_ls, label='Validation Loss', color='red', linestyle='-', marker='x')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    fig_name = save_path+'Data_'+str(if_data)+'_Rate_'+str(rate)+'_pid_'+str(pat_id)+'T_'+str(T)+'_EEGNet_'+'_Loss_pretraining.jpg'
    plt.savefig(fig_name)
    plt.plot()
    del train_dataloader
    gc.collect()
    #test
    print('Testing...')
    print('preparing data...')
    print(test_data.shape)
    print(test_label.shape)
    print('data prepared...')
    test_dataloader = DataLoader(test_subset, batch_size=16, shuffle=False, num_workers=2)
    test_class_weight = test_subset.class_weight
    test_sample_weight = test_subset.sample_weight
    print(' model testing...')
    teacher_test_precision, teacher_test_recall, teacher_test_f1, teacher_test_bca, teacher_test_spec,teacher_test_fpr,auc = bs_test(model, test_dataloader, device=device, modeltype='EEGNet', sample_weight = test_sample_weight)
    teacher_inference_time_sample = bs_test_infer(model, test_dataloader, device=device, modeltype='EEGNet')
    teacher_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(' Model #parameter: ', teacher_total_params)
    print(' Model current precision:', teacher_test_precision,' recall:',  teacher_test_recall, ' f1:', teacher_test_f1)
    print(' Model current BCA:',teacher_test_bca,' specificity:',  teacher_test_spec, ' fpr:', teacher_test_fpr)
    print(' Model inference time per sample:', teacher_inference_time_sample)
    print(' Model tested')

    # results.append([pat_id, rate, T, if_artifact, fold, teacher_test_precision, teacher_test_recall, teacher_test_f1, teacher_test_bca, teacher_test_spec,teacher_test_fpr, teacher_inference_time_sample, teacher_total_params])
    
    return

def exclewriter(pid, input_dim, rate_ls, if_data, dim_ls, if_artifact):
    results_p, results_tuned = [],[]
    for rate in rate_ls:
        for dim in dim_ls:
            results_folds = train_func(pid, rate, input_dim=input_dim, dim=dim, if_data = if_data, if_artifact=if_artifact)
            for i in results_folds:
                results_p.append(i)
    return results_p

def execute_tune(pid, input_dim, rate_ls, dim_ls, if_data=True, if_artifact=True):
    excel_file = str(if_artifact)+'_Specific_patient_training_Pat_'+pid+"_model_EEGNet_Data_"+str(if_data)+"_results.xlsx"
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:

        data_pre = exclewriter(pid, input_dim, rate_ls, if_data, dim_ls, if_artifact)
        # 转换为 Pandas DataFrame
        df_pre = pd.DataFrame(data_pre, columns=["Patient", "RATE", "T", 'artifact',"Dim","Precision", "Recall", "F1 Score", "BCA","Specificity","FPR",'inference_time_sample','total_params'])

        # 写入每个模型的结果到单独的 Sheet
        df_pre.to_excel(writer, sheet_name='EEGNet_spec', index=False)
    print(f"Results saved to {excel_file}")

if __name__ == '__main__':
    pid = '01'
    rate = 0

    train_func(pid, rate)