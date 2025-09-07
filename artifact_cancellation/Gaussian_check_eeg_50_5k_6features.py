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
import pickle
import scipy.io
from tqdm import tqdm
from sklearn.utils import resample
from models.EEGNet import EEGNetModel
from scipy.signal import butter, filtfilt
import scipy
import random
from scipy.signal import resample_poly

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


def hjorth_features_ndarray(data: np.ndarray) -> np.ndarray:
    """
    Compute Hjorth Activity, Mobility, and Complexity for each sample in a 2D ndarray.

    Parameters
    ----------
    data : np.ndarray
        2D array of shape (N, D), where N is the number of samples and D is the time series length.

    Returns
    -------
    np.ndarray
        2D array of shape (N, 3), where each row is [Activity, Mobility, Complexity] for the corresponding sample.
    """
    def compute_hjorth(X):
        activity = np.var(X)
        D = np.diff(X, prepend=X[0])  # prepend to keep length the same
        M2 = np.sum(D ** 2) / len(X)
        TP = np.sum(X ** 2)

        D2 = np.diff(D)
        M4 = np.sum(D2 ** 2) / len(X)

        mobility = np.sqrt(M2 / (TP + 1e-8))  # Adding a small value to avoid division by zero
        complexity = np.sqrt((M4 * TP) / (M2 ** 2 + 1e-8))
        return activity, mobility, complexity

    return np.array([compute_hjorth(sample) for sample in data])

def compute_features(x, fs=250, threshold_factor=3.0):
    """
    x: torch.Tensor or np.ndarray of shape (N, D)
    fs: sampling frequency (Hz)
    
    Returns:
        features: torch.Tensor of shape (N, 4)
    """
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x

    # 1. Line Length
    line_length = np.sum(np.abs(np.diff(x_np, axis=1)), axis=1)
    line_length = np.nan_to_num(line_length, nan=0.0)
    line_length = np.expand_dims(line_length, axis=1)
    
    # 2. Spike
    N, T = x.shape
    spike_counts = np.zeros(N, dtype=int)

    for i in range(N):
        signal = x[i]
        mean = np.mean(signal)
        std = np.std(signal)
        threshold = mean + threshold_factor * std

        # 一阶差分检测突变点（optional）
        # diff_signal = np.abs(np.diff(signal))

        # 简单阈值检测（超过上阈值或下阈值）
        spike_locs = np.where(np.abs(signal - mean) > threshold)[0]

        # 可选去重：避免连续多个点被算作多个 spike
        if len(spike_locs) > 0:
            clean_spikes = [spike_locs[0]]
            for j in range(1, len(spike_locs)):
                if spike_locs[j] - clean_spikes[-1] > 10:  # 至少间隔 10 个采样点
                    clean_spikes.append(spike_locs[j])
            spike_counts[i] = len(clean_spikes)

    spike_counts = np.expand_dims(spike_counts, axis=1)


    # 3. High Gamma Power (using Butterworth bandpass filter)
    def bandpower(signal, fs, band):
        b, a = butter(N=4, Wn=[band[0]/(fs/2), band[1]/(fs/2)], btype='band')
        filtered = filtfilt(b, a, signal)
        return np.mean(filtered ** 2, axis=1)

    gamma_power = bandpower(x_np, fs=fs, band=(30, 60))
    gamma_power = np.nan_to_num(gamma_power, nan=0.0)
    high_gamma_power = bandpower(x_np, fs=fs, band=(60, 100))
    high_gamma_power = np.nan_to_num(high_gamma_power, nan=0.0)
    hfo_power = np.stack([gamma_power, high_gamma_power], axis=1)

    # Concatenate features
    features = np.concatenate([line_length, spike_counts, hfo_power], axis=1)


    return torch.tensor(features, dtype=torch.float32)       
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
# seizure_data = np.load("SWEC_ETHZ_data/pat123_ALLclips_4s_seizure_data.npy")
# non_seizure_data = np.load("SWEC_ETHZ_data/pat123_ALLclips_4s_non_seizure_data.npy")
data_folder = "/net/inltitan1/scratch2/yuhxie/BrainBERT/"
non_seizure_data = np.load(data_folder + "eeg_train_dataall.npy")
# Set the parameters ###################################################################
# sample_freq = 2000
# clip_fs = 40
# nperseg = 1200
# noverlap = 1100
# test_first_layer = False
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
# path = "/home/ni/Documents/artifact-cancellation/performance-metrics/BrainBERT/"
# ckpt_path = path + "pretrained_weights/stft_large_pretrained.pth"
# cfg = OmegaConf.create({"upstream_ckpt": ckpt_path})
# model = build_model(cfg)
# model.to(device)
# init_state = torch.load(ckpt_path, map_location=device)
# load_model_weights(model, init_state['model'], False)

# # Evaluate FND ##############################################################################
# model.eval() # Set model to evaluation mode
      
Ns = [50,100,200,500,1000,5000]
Repeat_time = 10
cond = "non_seizure"  # or "non_seizure"
X_cond = non_seizure_data.reshape(-1,750)
X_cond = resample_poly(X_cond, up=512, down=250, axis=-1)
# # precompute full-dataset moments:
# full_feats = extract_all_features(X_cond)
# np.save("Gaussian_check_full_feats_"+cond+".npy", full_feats)
# mu_full = full_feats.mean(axis=0)
# var_full = full_feats.var(axis=0)

results_Hjorth = {"seizure": {'N': [], 'ks': [], 'mu': [], 'var': []},
        "non_seizure": {'N': [], 'ks': [], 'mu': [], 'var': []}}

results_LL = {"seizure": {'N': [], 'ks': [], 'mu': [], 'var': []},
        "non_seizure": {'N': [], 'ks': [], 'mu': [], 'var': []}}

results_Spike = {"seizure": {'N': [], 'ks': [], 'mu': [], 'var': []},
        "non_seizure": {'N': [], 'ks': [], 'mu': [], 'var': []}}

results_BP = {"seizure": {'N': [], 'ks': [], 'mu': [], 'var': []},
        "non_seizure": {'N': [], 'ks': [], 'mu': [], 'var': []}}

results_STFT = {"seizure": {'N': [], 'ks': [], 'mu': [], 'var': []},
        "non_seizure": {'N': [], 'ks': [], 'mu': [], 'var': []}}

results_all = {"seizure": {'N': [], 'ks': [], 'mu': [], 'var': []},
        "non_seizure": {'N': [], 'ks': [], 'mu': [], 'var': []}}


for N in Ns:
    print(f"Subsampling {N} samples from {cond} data")
    ks_list_Hjorth, mu_list_Hjorth, var_list_Hjorth = [], [], []
    ks_list_LL, mu_list_LL, var_list_LL = [], [], []
    ks_list_Spike, mu_list_Spike, var_list_Spike = [], [], []
    ks_list_BP, mu_list_BP, var_list_BP = [], [], []
    ks_list_STFT, mu_list_STFT, var_list_STFT = [], [], []
    ks_list_all, mu_list_all, var_list_all = [], [], []
    for r in range(Repeat_time):
        print(f"Repeat {r+1}/{Repeat_time}")
        X_sub = resample(X_cond, n_samples=N, replace=False)
        print(X_sub.shape)
        """Extract features."""
        # 1. Hjorth
        Hjorth_feats = hjorth_features_ndarray(X_sub)
        # 2. Line Length, Spike, High Gamma Power
        out_features = compute_features(X_sub, fs=sample_freq)
        line_length  = out_features[:, 0]
        spike       = out_features[:, 1]
        hfo_power    = out_features[:, 2:]
        # 3. STFT
        f_clean,t_clean,linear_clean = get_stft_stanford(X_sub, sample_freq, clip_fs=clip_fs, nperseg=nperseg, noverlap=noverlap, normalizing="zscore", return_onesided=True) #TODO hardcode sampling rate
        out_stft = linear_clean.reshape(linear_clean.shape[0], -1)
        # 4. all
        all_feats = np.concatenate((Hjorth_feats, out_features, out_stft), axis=1)
        
    
        mu_Hjorth = Hjorth_feats.mean(axis=0)          
        var_Hjorth = Hjorth_feats.var(axis=0)
        ks_stats_Hjorth = [kstest(Hjorth_feats[:,d], 'norm', args=(mu_Hjorth[d],var_Hjorth[d]**0.5)).statistic
                           for d in range(Hjorth_feats.shape[1])]
        ks_avg_Hjorth = np.mean(ks_stats_Hjorth)
        ks_list_Hjorth.append(ks_avg_Hjorth)
        mu_list_Hjorth.append(np.linalg.norm(mu_Hjorth))
        var_list_Hjorth.append(np.linalg.norm(var_Hjorth))
        
        mu_LL = line_length.mean(axis=0)
        var_LL = line_length.var(axis=0)
        ks_stats_LL = kstest(line_length, 'norm', args=(mu_LL, var_LL**0.5)).statistic
        ks_list_LL.append(ks_stats_LL)
        mu_list_LL.append(np.linalg.norm(mu_LL))
        var_list_LL.append(np.linalg.norm(var_LL))
        
        mu_Spike = spike.mean(axis=0)
        var_Spike = spike.var(axis=0)
        ks_stats_Spike = kstest(spike, 'norm', args=(mu_Spike, var_Spike**0.5)).statistic
        ks_list_Spike.append(ks_stats_Spike)
        mu_list_Spike.append(np.linalg.norm(mu_Spike))
        var_list_Spike.append(np.linalg.norm(var_Spike))
        
        
        mu_BP = hfo_power.mean(axis=0)
        var_BP = hfo_power.var(axis=0)
        ks_stats_BP = [kstest(hfo_power[:,d], 'norm', args=(mu_BP[d],var_BP[d]**0.5)).statistic
                      for d in range(hfo_power.shape[1])]
        ks_avg_BP = np.mean(ks_stats_BP)
        ks_list_BP.append(ks_avg_BP)
        mu_list_BP.append(np.linalg.norm(mu_BP))
        var_list_BP.append(np.linalg.norm(var_BP))
        
        
        mu_STFT = out_stft.mean(axis=0)
        var_STFT = out_stft.var(axis=0)
        ks_stats_STFT = [kstest(out_stft[:,d], 'norm', args=(mu_STFT[d],var_STFT[d]**0.5)).statistic
                         for d in range(out_stft.shape[1])]
        ks_avg_STFT = np.mean(ks_stats_STFT)
        ks_list_STFT.append(ks_avg_STFT)
        mu_list_STFT.append(np.linalg.norm(mu_STFT))
        var_list_STFT.append(np.linalg.norm(var_STFT))
        
        mu_all = all_feats.mean(axis=0)
        var_all = all_feats.var(axis=0)
        ks_stats_all = [kstest(all_feats[:,d], 'norm', args=(mu_all[d],var_all[d]**0.5)).statistic
                        for d in range(all_feats.shape[1])]
        ks_avg_all = np.mean(ks_stats_all)
        ks_list_all.append(ks_avg_all)
        mu_list_all.append(np.linalg.norm(mu_all))
        var_list_all.append(np.linalg.norm(var_all))
        
        
        

    # aggregate
    results_Hjorth[cond]['N'].append(N)
    results_Hjorth[cond]['ks'].append((np.mean(ks_list_Hjorth), np.std(ks_list_Hjorth)/np.sqrt(Repeat_time)))
    results_Hjorth[cond]['mu'].append((np.mean(mu_list_Hjorth), np.std(mu_list_Hjorth)/np.sqrt(Repeat_time)))
    results_Hjorth[cond]['var'].append((np.mean(var_list_Hjorth), np.std(var_list_Hjorth)/np.sqrt(Repeat_time)))
    np.save("FD_ValidationCheck/Gaussian_check_eeg_results_Hjorth_"+cond+"_N50_5k.npy", results_Hjorth)
    
    results_LL[cond]['N'].append(N)
    results_LL[cond]['ks'].append((np.mean(ks_list_LL), np.std(ks_list_LL)/np.sqrt(Repeat_time)))
    results_LL[cond]['mu'].append((np.mean(mu_list_LL), np.std(mu_list_LL)/np.sqrt(Repeat_time)))
    results_LL[cond]['var'].append((np.mean(var_list_LL), np.std(var_list_LL)/np.sqrt(Repeat_time)))
    np.save("FD_ValidationCheck/Gaussian_check_eeg_results_LL_"+cond+"_N50_5k.npy", results_LL)
    
    results_Spike[cond]['N'].append(N)
    results_Spike[cond]['ks'].append((np.mean(ks_list_Spike), np.std(ks_list_Spike)/np.sqrt(Repeat_time)))
    results_Spike[cond]['mu'].append((np.mean(mu_list_Spike), np.std(mu_list_Spike)/np.sqrt(Repeat_time)))
    results_Spike[cond]['var'].append((np.mean(var_list_Spike), np.std(var_list_Spike)/np.sqrt(Repeat_time)))
    np.save("FD_ValidationCheck/Gaussian_check_eeg_results_Spike_"+cond+"_N50_5k.npy", results_Spike)
    
    results_BP[cond]['N'].append(N)
    results_BP[cond]['ks'].append((np.mean(ks_list_BP), np.std(ks_list_BP)/np.sqrt(Repeat_time)))
    results_BP[cond]['mu'].append((np.mean(mu_list_BP), np.std(mu_list_BP)/np.sqrt(Repeat_time)))
    results_BP[cond]['var'].append((np.mean(var_list_BP), np.std(var_list_BP)/np.sqrt(Repeat_time)))
    np.save("FD_ValidationCheck/Gaussian_check_eeg_results_BP_"+cond+"_N50_5k.npy", results_BP)
    
    results_STFT[cond]['N'].append(N)
    results_STFT[cond]['ks'].append((np.mean(ks_list_STFT), np.std(ks_list_STFT)/np.sqrt(Repeat_time)))
    results_STFT[cond]['mu'].append((np.mean(mu_list_STFT), np.std(mu_list_STFT)/np.sqrt(Repeat_time)))
    results_STFT[cond]['var'].append((np.mean(var_list_STFT), np.std(var_list_STFT)/np.sqrt(Repeat_time)))
    np.save("FD_ValidationCheck/Gaussian_check_eeg_results_STFT_"+cond+"_N50_5k.npy", results_STFT)
    
    results_all[cond]['N'].append(N)
    results_all[cond]['ks'].append((np.mean(ks_list_all), np.std(ks_list_all)/np.sqrt(Repeat_time)))
    results_all[cond]['mu'].append((np.mean(mu_list_all), np.std(mu_list_all)/np.sqrt(Repeat_time)))
    results_all[cond]['var'].append((np.mean(var_list_all), np.std(var_list_all)/np.sqrt(Repeat_time)))
    np.save("FD_ValidationCheck/Gaussian_check_eeg_results_all_"+cond+"_N50_5k.npy", results_all)
    
    














    
    
