import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import models
from omegaconf import OmegaConf
import numpy as np
from scipy import signal, stats
import torch
from scipy.stats import kstest
from sklearn.utils import resample



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
        

def hjorth_batch(X):
    """
    Compute Hjorth mobility and complexity of multiple time series.

    Parameters
    ----------
    X : numpy.ndarray
        2D array of shape (N, T), where N is the number of samples and T is the length of each time series.

    Returns
    -------
    np.ndarray
        2D array of shape (N, 3) with Hjorth Activity, Mobility, and Complexity for each sample.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError("Input X must be a 2D numpy array of shape (N, T).")

    # Calculate Activity (variance of each time series)
    activity = np.var(X, axis=1)

    # Calculate first-order difference (D) for all samples
    D = np.diff(X, axis=1)
    
    # Pad the first difference for each sample (replicate the first value)
    D = np.concatenate([X[:, :1], D], axis=1) # insert X[0] at the beginning of D

    # Calculate Mobility
    M2 = np.mean(D ** 2, axis=1)
    TP = np.sum(X ** 2, axis=1) # = activity
    mobility = np.sqrt(M2 / TP)
    
    # Calculate second-order difference (for Complexity)
    D2 = np.diff(D, axis=1)
    M4 = np.mean(D2 ** 2, axis=1)
    complexity = np.sqrt((M4 * TP) / (M2 ** 2))

    # Combine results into a (N, 3) array
    hjorth_features = np.stack([activity, mobility, complexity], axis=1)
    
    return hjorth_features



# load the data ######################################################################
# seizure_data = np.load("SWEC_ETHZ_data/pat123_ALLclips_4s_seizure_data.npy")
non_seizure_data = np.load("SWEC_ETHZ_data/pat123_ALLclips_4s_non_seizure_data.npy")
# Set the parameters ###################################################################
sample_freq = 2000
# target_sampling_rate = 512
# print('GPU available:', torch.cuda.is_available())
# if torch.cuda.is_available():
#     device = torch.device("cuda:1")
#     print('Using GPU')
device = torch.device("cpu")
print('Using CPU')
# Evaluate FND ##############################################################################
 
Ns = [50,100,200,500,1000,5000]
Repeat_time = 10
cond = "non_seizure"  # or "non_seizure"
X_cond = non_seizure_data


# precompute full-dataset moments:
# X_full = X_cond
# """Extract features from the model."""
# f_clean,t_clean,linear_clean = get_stft_stanford(X_full, sample_freq, clip_fs=clip_fs, nperseg=nperseg, noverlap=noverlap, normalizing="zscore", return_onesided=True) 
# inputs_clean = torch.FloatTensor(linear_clean).transpose(1,2).to(device)
# print(inputs_clean.shape)
# mask_clean = torch.zeros((inputs_clean.shape[:2])).bool().to(device)
# with torch.no_grad():
#     F = model.forward(inputs_clean, mask_clean, intermediate_rep=True)

# F = F.cpu().numpy()
# F_avg_full = F.mean(axis=1)           # Shape: (1000, 768)
# mu_full = F_avg_full.mean(axis=0)
# var_full = F_avg_full.var(axis=0)
# # save mu and var
# np.save("mu_full_brainBERT.npy", mu_full)
# np.save("var_full_brainBERT.npy", var_full)

results = {"seizure": {'N': [], 'ks': [], 'mu': [], 'var': []},
        "non_seizure": {'N': [], 'ks': [], 'mu': [], 'var': []}}


for N in Ns:
    print(f"Subsampling {N} samples from {cond} data")
    ks_list, mu_list, var_list = [], [], []
    for r in range(Repeat_time):
        print(f"Repeat {r+1}/{Repeat_time}")
        X_sub = resample(X_cond, n_samples=N, replace=False)
        print(X_sub.shape)
        """Extract features from the model."""
        # apply Hjorth
        X_hjorth = hjorth_batch(X_sub) 
         
        mu_1 = X_hjorth.mean(axis=0)          # Shape: (3,) 
        print(mu_1.shape)
        var_1 = X_hjorth.var(axis=0)          # Shape: (3,) 
        
        # compute KS per-feature
        ks_stats_1 = []
        for d in range(X_hjorth.shape[1]):
            x = X_hjorth[:, d]
            if np.isnan(x).any() or np.isinf(x).any(): continue
            if np.std(x) < 1e-6: continue  # 样本太集中
            if var_1[d] < 1e-6: continue     # 理论分布也退化了
            ks = kstest(x, 'norm', args=(mu_1[d], var_1[d]**0.5)).statistic
            ks_stats_1.append(ks)
        ks_avg_1 = np.mean(ks_stats_1)
        ks_list.append(ks_avg_1)
        mu_list.append(np.linalg.norm(mu_1))
        var_list.append(np.linalg.norm(var_1))
        
    # aggregate
    print(ks_list)
    print(mu_list)
    print(var_list)
    results[cond]['N'].append(N)
    results[cond]['ks'].append((np.mean(ks_list), np.std(ks_list)/np.sqrt(Repeat_time)))
    results[cond]['mu'].append((np.mean(mu_list), np.std(mu_list)/np.sqrt(Repeat_time)))
    results[cond]['var'].append((np.mean(var_list), np.std(var_list)/np.sqrt(Repeat_time)))
    np.save("Gaussian_check_results_"+cond+"_Hjorth_N50_5k.npy", results)














    
    
