import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import models
from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
import torch
from scipy.linalg import sqrtm
from sklearn.manifold import TSNE
from scipy.stats import kstest
from sklearn.utils import resample
import pickle
import scipy.io
from tqdm import tqdm
from models.EEGNet import EEGNetModel
from scipy.signal import resample_poly


def load_selected_clips(file_path):
    """Load the selected clips from the pkl file."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def resample_signal(signal, original_rate, target_rate):
    """Resample signal to a new sampling rate."""
    num_original_samples = signal.shape[-1]
    num_target_samples = int(num_original_samples * target_rate / original_rate)
    
    if target_rate < original_rate:
        # Downsample by simple decimation
        factor = original_rate // target_rate
        return signal[:, ::factor]
    
    elif target_rate > original_rate:
        # Create the original and target time indices
        original_time = np.linspace(0, 1, num_original_samples, endpoint=False)
        target_time = np.linspace(0, 1, num_target_samples, endpoint=False)
        # Interpolate along the last axis (timesteps)
        interpolated_signal = np.zeros((signal.shape[0], num_target_samples))       
        for i in range(signal.shape[0]):  # Iterate over clips           
            interpolated_signal[i, :] = np.interp(target_time, original_time, signal[i, :])
        return interpolated_signal
    else: # Same sampling rate
        return signal
    
def resample_signal_3d(signal, original_rate, target_rate):
    """Resample signal to a new sampling rate."""
    num_original_samples = signal.shape[-1]
    num_target_samples = int(num_original_samples * target_rate / original_rate)
    
    if target_rate < original_rate:
        # Downsample by simple decimation
        factor = original_rate // target_rate
        return signal[:, :, ::factor]
    
    elif target_rate > original_rate:
        # Create the original and target time indices
        original_time = np.linspace(0, 1, num_original_samples, endpoint=False)
        target_time = np.linspace(0, 1, num_target_samples, endpoint=False)
        # Interpolate along the last axis (timesteps)
        interpolated_signal = np.zeros((signal.shape[0], signal.shape[1], num_target_samples))       
        for i in range(signal.shape[0]):  # Iterate over clips
            for j in range(signal.shape[1]):  # Iterate over channels
                interpolated_signal[i, j, :] = np.interp(target_time, original_time, signal[i, j, :])
        return interpolated_signal
    else: # Same sampling rate
        return signal
    
    
def get_stft_stanford(x, fs, clip_fs=-1, normalizing=None, **kwargs):
    f, t, Zxx = signal.stft(x, fs, **kwargs)
   
    Zxx = Zxx[:,:clip_fs,:]
    f = f[:clip_fs]

    Zxx = np.abs(Zxx)
    clip = 5 #To handle boundary effects
    if normalizing=="zscore":
        Zxx = Zxx[:,:,clip:-clip]
        Zxx = stats.zscore(Zxx, axis=-1)
        t = t[clip:-clip]
        Zxx=np.nan_to_num(Zxx,nan=0)

    if np.isnan(Zxx).any():
        import pdb; pdb.set_trace()

    return f, t, Zxx

    
def build_model(cfg):
    ckpt_path = cfg.upstream_ckpt
    init_state = torch.load(ckpt_path, map_location=device)
    upstream_cfg = init_state["model_cfg"]
    upstream = models.build_model(upstream_cfg)
    return upstream

def load_model_weights(model, states, multi_gpu):
    if multi_gpu:
        model.module.load_weights(states)
    else:
        model.load_weights(states)
        
# Compute Fréchet Distance using PyTorch
# def calculate_fid(mu1, sigma1, mu2, sigma2):
#     """Calculates the Fréchet distance between two multivariate Gaussians."""
#     diff = mu1 - mu2
#     eps = 1e-6
#     sigma1 += np.eye(sigma1.shape[0]) * eps
#     sigma2 += np.eye(sigma2.shape[0]) * eps
#     product = sigma1.dot(sigma2)

#     covmean, _ = sqrtm(product, disp=False)
    
#     # Numerical stability issues can cause imaginary component in covmean
#     if np.iscomplexobj(covmean):
#         covmean = covmean.real
    
#     fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
#     return fid



### Load the data ############################################################################
# data_folder = '/net/inltitan1/scratch2/Xiaoyong/Artifact_cancellation/SWEC_ETHZ_data/'
# file_path = data_folder + 'pat123_ALLclips_4s.pkl'
# sampling_rate_signal = 512  # Fixed sampling rate of the dataset
# target_sampling_rate = 2000  # Tunable resampling rate
# selected_data = load_selected_clips(file_path)
# # Count total number of samples for pre-allocation
# total_samples_seizure = 0
# total_samples_nonseizure = 0
# for patient_id in selected_data:
#     num_seizure_clips = len(selected_data[patient_id]['seizure_clips'])
#     num_non_seizure_clips = len(selected_data[patient_id]['non_seizure_clips'])
#     pat_channel_num = selected_data[patient_id]['seizure_clips'][0].shape[0]
#     total_samples_seizure += num_seizure_clips * pat_channel_num
#     total_samples_nonseizure += num_non_seizure_clips * pat_channel_num

# # Pre-allocate arrays
# timesteps =  selected_data[patient_id]['seizure_clips'][0].shape[1] 
# print("timesteps:", timesteps)
# seizure_data = np.zeros((total_samples_seizure, timesteps))
# non_seizure_data = np.zeros((total_samples_nonseizure, timesteps))

# # Fill in the arrays
# seizure_chan_idx, non_seizure_chan_idx = 0, 0  # track cumulative channel index
# for patient_id in selected_data:
#     for clip in selected_data[patient_id]['seizure_clips']:
#         channel_num = clip.shape[0]
#         seizure_data[seizure_chan_idx:seizure_chan_idx + channel_num, :] = clip.reshape(channel_num, -1)
#         seizure_chan_idx += channel_num  # cumulative

#     for clip in selected_data[patient_id]['non_seizure_clips']:
#         channel_num = clip.shape[0]
#         non_seizure_data[non_seizure_chan_idx:non_seizure_chan_idx + channel_num, :] = clip.reshape(channel_num, -1)
#         non_seizure_chan_idx += channel_num  # cumulative

# # resample the data to fit LNM
# seizure_data = resample_signal(seizure_data, original_rate=sampling_rate_signal, target_rate=target_sampling_rate)
# non_seizure_data = resample_signal(non_seizure_data, original_rate=sampling_rate_signal, target_rate=target_sampling_rate)
# print("non_seizure_data shape:", non_seizure_data.shape)
# np.save("SWEC_ETHZ_data/pat123_ALLclips_4s_seizure_data.npy", seizure_data)
# np.save("SWEC_ETHZ_data/pat123_ALLclips_4s_non_seizure_data.npy", non_seizure_data)

# load the data
# seizure_data = np.load("/net/inltitan1/scratch2/Xiaoyong/Artifact_cancellation/SWEC_ETHZ_data/pat123_ALLclips_4s_seizure_data.npy")
# non_seizure_data = np.load("/net/inltitan1/scratch2/Xiaoyong/Artifact_cancellation/SWEC_ETHZ_data/pat123_ALLclips_4s_non_seizure_data.npy")
non_seizure_data = np.load("eeg_train_dataall.npy")
print("load finished")
# Set the parameters ###################################################################
sample_freq = 250
clip_fs = 40
nperseg = 200
noverlap = 190
test_first_layer = False

## Load the model #########################################################################
print('GPU available:', torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device("cuda:1")
    print('Using GPU')
else:
    device = torch.device("cpu")
print('Using CPU')
path = "/home/ni/Documents/artifact-cancellation/performance-metrics/BrainBERT/"
ckpt_path = path + "pretrained_weights/stft_large_pretrained.pth"
cfg = OmegaConf.create({"upstream_ckpt": ckpt_path})
model = build_model(cfg)
model.to(device)
init_state = torch.load(ckpt_path, map_location=device)
load_model_weights(model, init_state['model'], False)

# Evaluate FND ##############################################################################
model.eval() # Set model to evaluation mode
      
Ns = [50,100,200,500,1000,5000]
Repeat_time = 10
cond = "non_seizure"  # or "non_seizure"
X_cond = non_seizure_data.reshape(-1,750)

X_cond = resample_poly(X_cond, up=512, down=250, axis=-1)

X_cond = X_cond.reshape(-1, 1, 1, X_cond.shape[-1])  # (454 * 88, 4, 512)

print(X_cond.shape)
# # precompute full-dataset moments:
# full_feats = extract_all_features(X_cond)
# np.save("Gaussian_check_full_feats_"+cond+".npy", full_feats)
# mu_full = full_feats.mean(axis=0)
# var_full = full_feats.var(axis=0)

results = {"seizure": {'N': [], 'ks': [], 'mu': [], 'var': []},
        "non_seizure": {'N': [], 'ks': [], 'mu': [], 'var': []}}

model = EEGNetModel(chans=1, classes=1, time_points=X_cond.shape[-1]).to(device)

ckpt_path = 'BrainBERT/checkpoints/save_models_True_data/rate_0/EEGNet/T_1/EEGNet_checkpoint_bcimulti.pt'

model.load_state_dict(torch.load(ckpt_path))
model.eval()
for N in Ns:
    print(f"Subsampling {N} samples from {cond} data")
    ks_list, mu_list, var_list = [], [], []
    for r in range(Repeat_time):
        print(f"Repeat {r+1}/{Repeat_time}")
        X_sub = resample(X_cond, n_samples=N, replace=False)
        print(X_sub.shape)
        # """Extract features from the model."""
        

        F_avg = model(torch.tensor(X_sub,dtype = torch.float32).to(device)).detach().cpu().numpy()
        
        # f_clean,t_clean,linear_clean = get_stft_stanford(X_sub, sample_freq, clip_fs=clip_fs, nperseg=nperseg, noverlap=noverlap, normalizing="zscore", return_onesided=True) 
        # inputs_clean = torch.FloatTensor(linear_clean).transpose(1,2).to(device)
        # # print(inputs_clean.shape)
        # mask_clean = torch.zeros((inputs_clean.shape[:2])).bool().to(device)
        # with torch.no_grad():
        #     F,_ = model.forward(inputs_clean, mask_clean, intermediate_rep=True)
        
        # F = F.cpu().numpy()
        
        # F_avg = F.mean(axis=1)           # Shape: (1000, 768)

        # F_avg = X_sub
        # F_avg = linear_clean.reshape(linear_clean.shape[0], -1)
        # F_avg = resample_poly(X_sub[:,:2000], up=512, down=2000, axis=-1)
        # F_avg = torch.tensor(F_avg, dtype = torch.float32).to(device)

        # F_avg = model(F_avg.unsqueeze(1).unsqueeze(1))
        # F_avg = F_avg.detach().cpu().numpy()
        print(F_avg.shape)
        # print(F_avg[0])
        mu = F_avg.mean(axis=0)          # Shape: (768,)
        var = F_avg.var(axis=0)          # Shape: (768,)
        # compute KS per-feature

        # ks_stats = [kstest(F_avg[:,d], 'norm', args=(mu[d],var[d]**0.5)).statistic
        #             for d in range(F_avg.shape[1])]
        ks_stats = []
        for d in range(F_avg.shape[1]):
            x = F_avg[:, d]
            if np.isnan(x).any() or np.isinf(x).any(): continue
            if np.std(x) < 1e-6: continue  # 样本太集中
            if var[d] < 1e-6: continue     # 理论分布也退化了
            ks = kstest(x, 'norm', args=(mu[d], var[d]**0.5)).statistic
            ks_stats.append(ks)

        ks_avg = np.mean(ks_stats)
        ks_list.append(ks_avg)
        mu_list.append(np.linalg.norm(mu))
        var_list.append(np.linalg.norm(var))
    # aggregate
    print(ks_list)
    print(mu_list)
    print(var_list)
    results[cond]['N'].append(N)
    results[cond]['ks'].append((np.mean(ks_list), np.std(ks_list)/np.sqrt(Repeat_time)))
    results[cond]['mu'].append((np.mean(mu_list), np.std(mu_list)/np.sqrt(Repeat_time)))
    results[cond]['var'].append((np.mean(var_list), np.std(var_list)/np.sqrt(Repeat_time)))
    np.save("Gaussian_check_eeg_results_"+cond+"_eegnetmulti_N50_5k.npy", results)














    
    