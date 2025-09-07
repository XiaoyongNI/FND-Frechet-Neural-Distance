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
        

def hjorth(X, D=None):
    """ Compute Hjorth mobility and complexity of a time series from either two
    cases below:
        1. X, the time series of type list (default)
        2. D, a first order differential sequence of X (if D is provided,
           recommended to speed up)

    In case 1, D is computed using Numpy's Difference function.

    Notes
    -----
    To speed up, it is recommended to compute D before calling this function
    because D may also be used by other functions whereas computing it here
    again will slow down.

    Parameters
    ----------

    X
        list

        a time series

    D
        list

        first order differential sequence of a time series

    Returns
    -------

    As indicated in return line

    Hjorth mobility and complexity

    """
    activity = np.var(X)
    if D is None:
        D = np.diff(X)
        D = D.tolist()

    D.insert(0, X[0])  # pad the first difference
    D = np.array(D)

    n = len(X)

    M2 = float(sum(D ** 2)) / n
    TP = sum(np.array(X) ** 2)
    M4 = 0
    for i in range(1, len(D)):
        M4 += (D[i] - D[i - 1]) ** 2
    M4 = M4 / n

    return activity, np.sqrt(M2 / TP), np.sqrt(
        float(M4) * TP / M2 / M2
    )  # Hjorth Activity, Mobility and Complexity



# load the data ######################################################################
# seizure_data = np.load("SWEC_ETHZ_data/pat123_ALLclips_4s_seizure_data.npy")
non_seizure_data = np.load("SWEC_ETHZ_data/pat123_ALLclips_4s_non_seizure_data.npy")
# Set the parameters ###################################################################
sample_freq = 2000
clip_fs = 40
nperseg = 1200
noverlap = 1100
test_first_layer = False

## Load the model #########################################################################
# print('GPU available:', torch.cuda.is_available())
# if torch.cuda.is_available():
#     device = torch.device("cuda:1")
#     print('Using GPU')
# else:
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
      
Ns = [10000, 20000, 50000, 100000]
Repeat_time = 5
cond = "non_seizure"  # or "non_seizure"
X_cond = non_seizure_data


# precompute full-dataset moments:
X_full = X_cond
"""Extract features from the model."""
f_clean,t_clean,linear_clean = get_stft_stanford(X_full, sample_freq, clip_fs=clip_fs, nperseg=nperseg, noverlap=noverlap, normalizing="zscore", return_onesided=True) 
inputs_clean = torch.FloatTensor(linear_clean).transpose(1,2).to(device)
print(inputs_clean.shape)
mask_clean = torch.zeros((inputs_clean.shape[:2])).bool().to(device)
with torch.no_grad():
    F = model.forward(inputs_clean, mask_clean, intermediate_rep=True)

F = F.cpu().numpy()
F_avg_full = F.mean(axis=1)           # Shape: (1000, 768)
mu_full = F_avg_full.mean(axis=0)
var_full = F_avg_full.var(axis=0)
# save mu and var
np.save("mu_full_brainBERT.npy", mu_full)
np.save("var_full_brainBERT.npy", var_full)

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
        f_clean,t_clean,linear_clean = get_stft_stanford(X_sub, sample_freq, clip_fs=clip_fs, nperseg=nperseg, noverlap=noverlap, normalizing="zscore", return_onesided=True) 
        inputs_clean = torch.FloatTensor(linear_clean).transpose(1,2).to(device)
        print(inputs_clean.shape)
        mask_clean = torch.zeros((inputs_clean.shape[:2])).bool().to(device)
        with torch.no_grad():
            F = model.forward(inputs_clean, mask_clean, intermediate_rep=True)
        
        F = F.cpu().numpy()
        
        F_avg = F.mean(axis=1)           # Shape: (1000, 768)
        mu = F_avg.mean(axis=0)          # Shape: (768,)
        var = F_avg.var(axis=0)          # Shape: (768,)
        # compute KS per-feature
        ks_stats = [kstest(F_avg[:,d], 'norm', args=(mu[d],var[d]**0.5)).statistic
                    for d in range(F_avg.shape[1])]
        ks_avg = np.mean(ks_stats)
        ks_list.append(ks_avg)
        mu_list.append(np.linalg.norm(mu))
        var_list.append(np.linalg.norm(var))
    # aggregate
    results[cond]['N'].append(N)
    results[cond]['ks'].append((np.mean(ks_list), np.std(ks_list)/np.sqrt(Repeat_time)))
    results[cond]['mu'].append((np.mean(mu_list), np.std(mu_list)/np.sqrt(Repeat_time)))
    results[cond]['var'].append((np.mean(var_list), np.std(var_list)/np.sqrt(Repeat_time)))
    np.save("Gaussian_check_results_"+cond+"_N10k_100k.npy", results)














    
    
