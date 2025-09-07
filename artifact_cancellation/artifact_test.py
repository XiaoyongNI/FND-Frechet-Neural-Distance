import sys

from pyparsing import C
print(sys.path)
import numpy as np
import pandas as pd
import scipy.io as scio

from models.EEGNet import EEGNetModel
from models.Chrononet import ChronoNet
from models.GRU import MultiChannelGRU

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






T=1

model = EEGNetModel(chans=88, classes=1, time_points=int(T*512)).to(device)
# model = ChronoNet().to(device)
# model = MultiChannelGRU(input_size = 512, hidden_size=32, num_layers=2, output_size=1, T=T, channels= 88,device=device).to(device)

ckpt_path = 'BrainBERT/checkpoints/save_models_True_data/rate_0/EEGNet/T_1/pat_01_EEGNet_checkpoint.pt'

model.load_state_dict(torch.load(ckpt_path))
model.eval()


code_path = '/net/inltitan1/scratch2/Xiaoyong/Artifact_cancellation/'
cases = ["artifact","ASAR", "interpolation", "SVD_acrossChannels"]
path_artifact = [code_path+"ethz_data/mixed_test_X_pat1_f512_amp73.npy", code_path+"ethz_data/ASAR_amp73.npy",code_path+"ethz_data/interpolation_amp73.npy",code_path+"ethz_data/SVD_AcrossChannels_N1_amp73.npy"]

for i in range(len(path_artifact)):
    print("Testing case: ", cases[i])
    test_x = np.load(path_artifact[i])

    # print(test_x[0,:,0])
    
    path = '/net/inltitan1/scratch2/yuhxie/ethz_data/'

    test_y = path + 'mixed_test_Y_pat1.npy'

    # test_x = np.load(test_x)
    test_y = np.load(test_y)

    test_x = torch.tensor(test_x, dtype=torch.float32)
    test_y = torch.tensor(test_y)

    print(test_x.shape, test_y.shape)

    test_x = test_x.unsqueeze(2)

    test_subset = myDataset.iEEG_Dataset(test_x, test_y, device=device)

    channel = test_x.shape[1]
    feature = test_x.shape[3]
    print('channel: ', channel)
    print('feature: ', feature)



    test_dataloader = DataLoader(test_subset, batch_size=16, shuffle=False, num_workers=2)


    test_class_weight = test_subset.class_weight
    test_sample_weight = test_subset.sample_weight
    print(' model testing...')
    teacher_test_precision, teacher_test_recall, teacher_test_f1, teacher_test_bca, teacher_test_spec,teacher_test_fpr, auc, acc = bs_test(model, test_dataloader, device=device, modeltype='EEGNet', sample_weight = test_sample_weight)
    teacher_inference_time_sample = bs_test_infer(model, test_dataloader, device=device, modeltype='EEGNet')
    teacher_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(' Model #parameter: ', teacher_total_params)
    print(' Model current precision:', teacher_test_precision,' recall:',  teacher_test_recall, ' f1:', teacher_test_f1, 'auc:', auc, 'acc:', acc)
    print(' Model current BCA:',teacher_test_bca,' specificity:',  teacher_test_spec, ' fpr:', teacher_test_fpr)
    print(' Model inference time per sample:', teacher_inference_time_sample)
    print(' Model tested')
