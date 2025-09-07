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
# path_clean = "/net/inltitan1/scratch2/yuhxie/ethz_data/123_mixed_seizure1_amp57.npy"
path_clean = "/net/inltitan1/scratch2/Xiaoyong/Artifact_cancellation/ethz_data/ASAR_amp73.npy"
wav_clean = np.load(path_clean)
# upsample from 512Hz to 2000Hz
wav_clean = resample(wav_clean, 2000, axis=2) 
wav_clean = wav_clean.reshape(-1, 2000)

print(wav_clean.shape)

f_clean,t_clean,linear_clean = get_stft_stanford(wav_clean, sample_freq, clip_fs=clip_fs, nperseg=nperseg, noverlap=noverlap, normalizing="zscore", return_onesided=True) #TODO hardcode sampling rate
inputs_clean = torch.FloatTensor(linear_clean).transpose(1,2).to(device)
print(inputs_clean.shape)
mask_clean = torch.zeros((inputs_clean.shape[:2])).bool().to(device)
with torch.no_grad():
    firstlayer_clean, out_clean = model.forward(inputs_clean, mask_clean, intermediate_rep=True)

# test cases
# cases = ["ASAR", "interpolation", "SVD_acrossChannels", "SVD_acrossPulses"]
# path_artifact = ["ethz_data/ASAR.npy","ethz_data/interpolation.npy","ethz_data/SVD_AcrossChannels_N1.npy","ethz_data/SVD_AcrossPulses_N1.npy"]
cases = ["NC", "SC", "NA", "SA"]
path_artifact = ["/net/inltitan1/scratch2/yuhxie/ethz_data/123_clean_nonseizure2_amp57.npy","/net/inltitan1/scratch2/yuhxie/ethz_data/123_clean_seizure2_amp57.npy","/net/inltitan1/scratch2/yuhxie/ethz_data/123_mixed_nonseizure2_amp57.npy","/net/inltitan1/scratch2/yuhxie/ethz_data/123_mixed_seizure2_amp57.npy"]

for i in range(len(path_artifact)):
    print("Testing case: ", cases[i])
    wav_artifact = np.load(path_artifact[i])
    # upsample from 512Hz to 2000Hz
    # wav_artifact = resample(wav_artifact, 2000, axis=2) 
    wav_artifact = wav_artifact.reshape(-1, 2000)

    f_artifact,t_artifact,linear_artifact = get_stft_stanford(wav_artifact, sample_freq, clip_fs=clip_fs, nperseg=nperseg, noverlap=noverlap, normalizing="zscore", return_onesided=True) #TODO hardcode sampling rate
    inputs_artifact = torch.FloatTensor(linear_artifact).transpose(1,2).to(device)
    mask_artifact = torch.zeros((inputs_artifact.shape[:2])).bool().to(device)
    with torch.no_grad():
        firstlayer_artifact, out_artifact = model.forward(inputs_artifact, mask_artifact, intermediate_rep=True)


    ### FND
    # Flatten last two dimensions (combine the 40 and 768)

    print(firstlayer_clean.shape, out_clean.shape)

    if test_first_layer:
        print("test first layer")
        firstlayer_clean = firstlayer_clean.mean(dim=1)
        firstlayer_artifact = firstlayer_artifact.mean(dim=1)
        out_clean_flattened = firstlayer_clean.reshape(firstlayer_clean.shape[0], -1).cpu().numpy()
        out_artifact_flattened = firstlayer_artifact.reshape(firstlayer_artifact.shape[0], -1).cpu().numpy()
    else:
        print("test last layer")
        print(out_clean.shape)
        # out_clean = out_clean.mean(dim=1)
        # out_artifact = out_artifact.mean(dim=1)
        out_clean_flattened = out_clean.reshape(out_clean.shape[0], -1).cpu().numpy()
        out_artifact_flattened = out_artifact.reshape(out_artifact.shape[0], -1).cpu().numpy()

    print(out_clean_flattened.shape)

    # visualize_tsne(out_clean_flattened, out_artifact_flattened, label1 = 'seizure', label2 = 'nonseizure', layer = 'first' if test_first_layer else 'last')

    # Calculate the means and covariances of the encoded datasets
    mu_clean = np.mean(out_clean_flattened, axis=0)
    sigma_clean = np.cov(out_clean_flattened, rowvar=False)

    mu_artifact = np.mean(out_artifact_flattened, axis=0)
    sigma_artifact = np.cov(out_artifact_flattened, rowvar=False)

    # Compute the Fréchet Neural Distance
    fid = calculate_fid(mu_clean, sigma_clean, mu_artifact, sigma_artifact)
    print(f"Fréchet Neural Distance (FND): {fid}")