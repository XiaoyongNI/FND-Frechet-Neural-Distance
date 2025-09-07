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
from scipy.signal import resample
from models.EEGNet import EEGNetModel
from scipy.signal import resample_poly
from scipy.signal import butter, filtfilt
import scipy
import random

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

def compute_features(x, fs=1000, gamma_band=(70, 150)):
    """
    x: torch.Tensor or np.ndarray of shape (N, D)
    fs: sampling frequency (Hz)
    gamma_band: tuple defining high gamma band (Hz)
    
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

    # 2. Skewness
    skewness = scipy.stats.skew(x_np, axis=1)
    skewness = np.nan_to_num(skewness, nan=0.0)

    # 3. Kurtosis
    kurtosis = scipy.stats.kurtosis(x_np, axis=1)
    kurtosis = np.nan_to_num(kurtosis, nan=0.0)

    # 4. High Gamma Power (using Butterworth bandpass filter)
    def bandpower(signal, fs, band):
        b, a = butter(N=4, Wn=[band[0]/(fs/2), band[1]/(fs/2)], btype='band')
        filtered = filtfilt(b, a, signal)
        return np.mean(filtered ** 2, axis=1)

    hfo_power = bandpower(x_np, fs=fs, band=gamma_band)
    hfo_power = np.nan_to_num(hfo_power, nan=0.0)

    # Stack into N×4 feature matrix
    features = np.stack([line_length, skewness, kurtosis, hfo_power], axis=1)

    return torch.tensor(features, dtype=torch.float32)


def plot_time_series_stanford(path, sample_rate):
    wav = np.load(path)
    wav = wav[10,:]
    plt.figure(figsize=(10,3))
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=20)
    plt.ylabel(u"Voltage (\u03bcV)", fontsize=25)
    plt.xticks(np.arange(0,len(wav)+1, sample_rate), [x/sample_rate for x in np.arange(0,len(wav)+1, sample_rate)])
    plt.xlabel("Time (ms)", fontsize=25)
    plt.plot(wav)
    

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
        Zxx = np.nan_to_num(Zxx, nan=0)  # nan=0.0 or 1e6 depending on intent

    if np.isnan(Zxx).any():
        import pdb; pdb.set_trace()

    return f, t, Zxx

def plot_stft(path):
    wav = np.load(path)
    f,t,linear = get_stft_stanford(wav, sample_freq, clip_fs=clip_fs, nperseg=nperseg, noverlap=noverlap, normalizing="zscore", return_onesided=True) #TODO hardcode sampling rate
    plt.figure(figsize=(15,3))
    # f[-1]=200
    g1 = plt.pcolormesh(t,f,np.squeeze(linear[1,:,:]), shading="gouraud", vmin=-3.5, vmax=6)

    cbar = plt.colorbar(g1)
    tick_font_size = 15
    cbar.ax.tick_params(labelsize=tick_font_size)
    cbar.ax.set_ylabel("Power (Arbitrary units)", fontsize=15)
    plt.xticks(fontsize=20)
    plt.ylabel("")
    plt.yticks(fontsize=20)
    plt.xlabel("Time (s)", fontsize=20)
    plt.ylabel("Frequency (Hz)", fontsize=20)
    
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
def calculate_fid(mu1, sigma1, mu2, sigma2):
    """Calculates the Fréchet distance between two multivariate Gaussians."""
    diff = mu1 - mu2
    eps = 1e-6
    sigma1 += np.eye(sigma1.shape[0]) * eps
    sigma2 += np.eye(sigma2.shape[0]) * eps
    product = sigma1.dot(sigma2)

    covmean, _ = sqrtm(product, disp=False)
    
    # Numerical stability issues can cause imaginary component in covmean
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def visualize_tsne(clean_feat: np.ndarray, artifact_feat: np.ndarray, title="t-SNE Visualization", label1 = 'clean', label2 = 'artifact', layer = 'first'):
    """
    clean_feat: np.ndarray of shape (N, D) - features without artifact
    artifact_feat: np.ndarray of shape (N, D) - features with artifact
    """
    # 合并特征
    features = np.concatenate([clean_feat, artifact_feat], axis=0)  # [2N, D]
    labels = np.array([0] * len(clean_feat) + [1] * len(artifact_feat))  # 0: clean, 1: artifact

    # t-SNE 降维
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    features_2d = tsne.fit_transform(features)

    # 可视化
    plt.figure(figsize=(8, 6))
    plt.scatter(
        features_2d[labels == 0, 0], features_2d[labels == 0, 1],
        c='blue', label=label1, alpha=0.6
    )
    plt.scatter(
        features_2d[labels == 1, 0], features_2d[labels == 1, 1],
        c='red', label=label2, alpha=0.6
    )
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("tsne_visualization_"+label1+"_"+label2+"_layer_"+layer+".png")

# Set the parameters
sample_freq = 2000
clip_fs = 40
nperseg = 1200
noverlap = 1100
test_first_layer = False

## Load the model
# print('GPU available:', torch.cuda.is_available())
# if torch.cuda.is_available():
#     device = torch.device("cuda")
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

model.eval() # Set model to evaluation mode

model = EEGNetModel(chans=1, classes=1, time_points=512).to(device)

ckpt_path = 'BrainBERT/checkpoints/save_models_True_data/rate_0/EEGNet/T_1/pat_01_EEGNet_checkpoint_singlechannel2.pt'

model.load_state_dict(torch.load(ckpt_path))
model.eval()

path_clean = "/net/inltitan1/scratch2/yuhxie/ethz_data/mixed_train_X_pat1.npy"
wav_clean = np.load(path_clean)
# upsample from 512Hz to 2000Hz
random.seed(42)
sampled = random.sample(range(8340), 200)

print(wav_clean[sampled[:100]].shape)

wav_clean2 = wav_clean[sampled[100:]]

upsampled_wav = resample(wav_clean[sampled[:100]], 2000, axis=2) 
wav_clean = upsampled_wav.reshape(-1, 2000)




# wav_clean = resample_poly(wav_clean, up=512, down=2000, axis=-1)

print(wav_clean.shape)

f_clean,t_clean,linear_clean = get_stft_stanford(wav_clean, sample_freq, clip_fs=clip_fs, nperseg=nperseg, noverlap=noverlap, normalizing="zscore", return_onesided=True) #TODO hardcode sampling rate
inputs_clean = torch.FloatTensor(linear_clean).transpose(1,2).to(device)
# print(inputs_clean.shape)

# mask_clean = torch.zeros((inputs_clean.shape[:2])).bool().to(device)
# with torch.no_grad():
#     firstlayer_clean, out_clean = model.forward(inputs_clean, mask_clean, intermediate_rep=True)

# out_clean = linear_clean
out_clean_model = model(torch.tensor(resample_poly(wav_clean, up=512, down=2000, axis=-1), dtype = torch.float32).unsqueeze(1).unsqueeze(1))
# print(out_clean_model.shape)
out_clean_stft = linear_clean.reshape(linear_clean.shape[0], -1)
out_clean_hjorth = hjorth_features_ndarray(wav_clean)
out_clean_features = compute_features(wav_clean, fs=2000, gamma_band=(70, 150))
out_clean_flattened = np.concatenate((out_clean_stft, out_clean_hjorth, out_clean_features), axis=1)
out_clean_raw = wav_clean
# test cases
code_path = '/net/inltitan1/scratch2/Xiaoyong/Artifact_cancellation/'
cases = ["artifact","ASAR", "interpolation", "SVD_acrossChannels", "SVD_acrossPulses"]
path_artifact = [code_path+"ethz_data/mixed_test_X_pat1_f512_amp73.npy", code_path+"ethz_data/ASAR_amp73.npy",code_path+"ethz_data/interpolation_amp73.npy",code_path+"ethz_data/SVD_AcrossChannels_N1_amp73.npy",code_path+"ethz_data/SVD_AcrossPulses_N1_amp73.npy"]

for i in range(len(path_artifact)):
    print("Testing case: ", cases[i])
    wav_artifact = np.load(path_artifact[i])
    # upsample from 512Hz to 2000Hz
    upsampled_wav_artifact = resample(wav_artifact[random.sample(range(4170), 100)], 2000, axis=2) 
    wav_artifact = upsampled_wav_artifact.reshape(-1, 2000)
    # wav_artifact = resample_poly(wav_artifact, up=512, down=2000, axis=-1)

    f_artifact,t_artifact,linear_artifact = get_stft_stanford(wav_artifact, sample_freq, clip_fs=clip_fs, nperseg=nperseg, noverlap=noverlap, normalizing="zscore", return_onesided=True) #TODO hardcode sampling rate
    
    # out_artifact = linear_artifact
    # inputs_artifact = torch.FloatTensor(linear_artifact).transpose(1,2).to(device)
    # mask_artifact = torch.zeros((inputs_artifact.shape[:2])).bool().to(device)
    # with torch.no_grad():
    #     firstlayer_artifact, out_artifact = model.forward(inputs_artifact, mask_artifact, intermediate_rep=True)

    # out_artifact = hjorth_features_ndarray(wav_artifact)
    out_artifact_model = model(torch.tensor(resample_poly(wav_artifact, up=512, down=2000, axis=-1), dtype = torch.float32).unsqueeze(1).unsqueeze(1))
    
    ### FND
    # Flatten last two dimensions (combine the 40 and 768)

    out_artifact_stft = linear_artifact.reshape(linear_artifact.shape[0], -1)
    out_artifact_hjorth = hjorth_features_ndarray(wav_artifact)
    out_artifact_features = compute_features(wav_artifact, fs=2000, gamma_band=(70, 150))

    out_artifact_flattened = np.concatenate((out_artifact_stft, out_artifact_hjorth, out_artifact_features), axis=1)
    out_artifact_raw = wav_artifact

    # print(firstlayer_clean.shape, out_clean.shape)

    # if test_first_layer:
    #     print("test first layer")
    #     firstlayer_clean = firstlayer_clean.mean(dim=1)
    #     firstlayer_artifact = firstlayer_artifact.mean(dim=1)
    #     out_clean_flattened = firstlayer_clean.reshape(firstlayer_clean.shape[0], -1).cpu().numpy()
    #     out_artifact_flattened = firstlayer_artifact.reshape(firstlayer_artifact.shape[0], -1).cpu().numpy()
    # else:
    #     print("test last layer")
    #     print(out_clean.shape)
    #     # out_clean = out_clean.mean(dim=1)
    #     # out_artifact = out_artifact.mean(dim=1)
    #     out_clean_flattened = out_clean.reshape(out_clean.shape[0], -1).detach().cpu().numpy()
    #     out_artifact_flattened = out_artifact.reshape(out_artifact.shape[0], -1).detach().cpu().numpy()

    # print(out_clean_flattened.shape)

    # visualize_tsne(out_clean_flattened, out_artifact_flattened, label1 = 'seizure', label2 = 'nonseizure', layer = 'first' if test_first_layer else 'last')

    # Calculate the means and covariances of the encoded datasets
    mu_clean = np.mean(out_clean_flattened, axis=0)
    sigma_clean = np.cov(out_clean_flattened, rowvar=False)

    mu_artifact = np.mean(out_artifact_flattened, axis=0)
    sigma_artifact = np.cov(out_artifact_flattened, rowvar=False)

    # Compute the Fréchet Neural Distance
    fid = calculate_fid(mu_clean, sigma_clean, mu_artifact, sigma_artifact)
    print(f"Fréchet Neural Distance Concat (FND): {fid}")

    mu_clean = np.mean(out_clean_model.reshape(out_clean_model.shape[0],-1).detach().cpu().numpy(), axis=0)
    sigma_clean = np.cov(out_clean_model.reshape(out_clean_model.shape[0],-1).detach().cpu().numpy(), rowvar=False)

    mu_artifact = np.mean(out_artifact_model.reshape(out_artifact_model.shape[0],-1).detach().cpu().numpy(), axis=0)
    sigma_artifact = np.cov(out_artifact_model.reshape(out_artifact_model.shape[0],-1).detach().cpu().numpy(), rowvar=False)

    # Compute the Fréchet Neural Distance
    fid = calculate_fid(mu_clean, sigma_clean, mu_artifact, sigma_artifact)
    print(f"Fréchet Neural Distance FSTD (FND): {fid}")

    mse = ((out_clean_flattened - out_artifact_flattened) ** 2).mean()
    print(f"Mean Squared Error concat (MSE): {mse}")

    mse = ((out_clean_stft - out_artifact_stft) ** 2).mean()
    print(f"Mean Squared Error stft (MSE): {mse}")

    mse = ((out_clean_features - out_artifact_features) ** 2).mean(dim = 0)
    print(f"Mean Squared Error Features (MSE): {mse}")

    mse = ((out_clean_raw - out_artifact_raw) ** 2).mean()
    print(f"Mean Squared Error RAW (MSE): {mse}")

    # break