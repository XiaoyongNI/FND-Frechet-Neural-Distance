import sys
print(sys.path)
import numpy as np
import pandas as pd
import scipy.io as scio

from models.EEGNet import EEGNetModel

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
from scipy.signal import resample_poly

torch.manual_seed(42)
pids = ['01','02','03','04','05', '06', '07', '08','09','10','11','12','13','14','15','16','17','18']
GPU_ID = 0
device = torch.device('cuda:{}'.format(GPU_ID) if torch.cuda.is_available() else "cpu")
print(device)
batch_size = 256


def train_func(pat_id, rate, device = device, step=2, dim=32,if_data=True, if_artifact=True):


    X_train = np.load('eeg_train_dataall.npy')
    X_test = np.load('eeg_test_dataall.npy')
    y_train = np.load('eeg_train_labelall.npy')
    y_test = np.load('eeg_test_labelall.npy')

    X_train = resample_poly(X_train, up=512, down=250, axis=-1)
    X_test = resample_poly(X_test, up=512, down=250, axis=-1)

    X_train = X_train.reshape(-1, 1, 1, X_train.shape[-1])  # (454 * 88, 4, 512)
    X_test = X_test.reshape(-1, 1, 1, X_test.shape[-1])  # (454 * 88, 4, 512)
    y_train = np.repeat(y_train, 22)  # (454 * 88,)
    y_test = np.repeat(y_test, 22)  # (454 * 88,)



    train_data, val_data, train_label, val_label = train_test_split(X_train, y_train,  test_size=0.2, stratify=y_train, random_state=42)
    train_data, val_data, test_data = torch.tensor(train_data,dtype = torch.float32), torch.tensor(val_data,dtype = torch.float32), torch.tensor(X_test,dtype = torch.float32)
    train_label, val_label, test_label = torch.tensor(train_label,dtype = torch.float32), torch.tensor(val_label,dtype = torch.float32),torch.tensor(y_test,dtype = torch.float32)
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

    model = EEGNetModel(chans=int(channel), classes=1, time_points=int(feature)).to(device)
    ckpt_path = 'BrainBERT/checkpoints/save_models_True_data/rate_0/EEGNet/T_1/EEGNet_checkpoint_bcisingle.pt'

    model.load_state_dict(torch.load(ckpt_path))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, betas=(0.9, 0.999))
    criterion =  nn.BCEWithLogitsLoss()
    save_path = 'BrainBERT/checkpoints/save_models_'+str(if_data)+'_data/rate_'+str(rate)+'/EEGNet/'+'T_'+str(1)+'/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    early_stopping = EarlyStopping(patience=15, verbose=True, path=os.path.join(save_path, 'EEGNet_checkpoint_bcimulti.pt'))
    start_epoch = 0
    train_ls, val_ls = [], []
    for epoch in range(start_epoch, 5):
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
    fig_name = save_path+'_Loss_pretraining.jpg'
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
    teacher_test_precision, teacher_test_recall, teacher_test_f1, teacher_test_bca, teacher_test_spec,teacher_test_fpr = bs_test(model, test_dataloader, device=device, modeltype='EEGNet', sample_weight = test_sample_weight)
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