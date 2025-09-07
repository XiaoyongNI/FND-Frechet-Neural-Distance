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
batch_size = 16

path = '/net/inltitan1/scratch2/yuhxie/ethz_data/'

test_x = path + 'mixed_test_X_pat1.npy'
test_y = path + 'mixed_test_Y_pat1.npy'

test_x = np.load(test_x)
test_y = np.load(test_y)


unique_values, counts = np.unique(test_y, return_counts=True)

# 打印结果
print("不同值：", unique_values)
print("对应出现次数：", counts)

# test_x = resample_poly(test_x, 32, 125, axis=2)

# factor = 2000 // 512
# test_x = test_x[:, :, ::factor]

signal = test_x[0,0,:]  # shape: (1, 2000)

# 画出这条信号曲线
plt.figure(figsize=(12, 4))
plt.plot(signal, color='blue')  # signal[0] 是 1D 向量
plt.title("1×2000 Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.savefig("1x2000_signal.png")

test_x = torch.tensor(test_x, dtype=torch.float32)
test_y = torch.tensor(test_y)




print(test_x.shape, test_y.shape)

test_x = test_x.unsqueeze(2)

test_subset = myDataset.iEEG_Dataset(test_x, test_y, device=device)

channel = test_x.shape[1]
feature = test_x.shape[3]
print('channel: ', channel)
print('feature: ', feature)

T=1

model = EEGNetModel(chans=int(channel), classes=1, time_points=int(T*feature)).to(device)

ckpt_path = 'BrainBERT/checkpoints/save_models_True_data/rate_0/EEGNet/T_1/pat_01_EEGNet_checkpoint_nonorm.pt'

model.load_state_dict(torch.load(ckpt_path))
model.eval()

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
