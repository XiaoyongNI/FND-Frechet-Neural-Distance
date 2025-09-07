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
test_first_layer = True

## Load the model
print('GPU available:', torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device("cuda")
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

model.eval() # Set model to evaluation mode
path_clean = "ethz_data/clean_nonseizure1.npy"
wav_clean = np.load(path_clean)
wav_clean = wav_clean.reshape(-1, 2000)

print(wav_clean.shape)

path_artifact = "ethz_data/clean_nonseizure2.npy"
# path_artifact = "datasets/stanford-dataset/svd_window_size/adaptive N(0.6)/artifact_alltrials_allch_svdNadap_window90.npy"
# path_artifact = "datasets/stanford-dataset/baselines/artifact_alltrials_allch_asar.npy"

wav_artifact = np.load(path_artifact)
wav_artifact = wav_artifact.reshape(-1, 2000)



### FND
# Flatten last two dimensions (combine the 40 and 768)


out_clean_flattened = wav_clean
out_artifact_flattened = wav_artifact

# visualize_tsne(out_clean_flattened, out_artifact_flattened, label1 = 'seizure', label2 = 'nonseizure', layer = 'first' if test_first_layer else 'last')

# Calculate the means and covariances of the encoded datasets
MSE = np.mean((out_clean_flattened - out_artifact_flattened) ** 2, axis=0)
print(f"MSE: {MSE.mean()}")