import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from omegaconf import OmegaConf
import numpy as np
import torch
from scipy.linalg import sqrtm
from scipy.signal import resample

from datetime import datetime
import argparse
from utils.utils import _logger
from model import TFC
from mne.decoding import CSP

# Args selections
start_time = datetime.now()
parser = argparse.ArgumentParser()

home_dir = os.getcwd()
parser.add_argument('--run_description', default='run1', type=str, help='Experiment Description')
parser.add_argument('--seed', default=2023, type=int, help='seed value')

parser.add_argument('--training_mode', default='pre_train', type=str, help='pre_train, fine_tune')
parser.add_argument('--pretrain_dataset', default='SleepEEG', type=str,
                    help='Dataset of choice: SleepEEG, FD_A, HAR, ECG')
parser.add_argument('--target_dataset', default='Epilepsy', type=str,
                    help='Dataset of choice: Epilepsy, FD_B, Gesture, EMG')

parser.add_argument('--logs_save_dir', default='experiments_logs', type=str, help='saving directory')
parser.add_argument('--device', default='cuda', type=str, help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str, help='Project home directory')
parser.add_argument('--subset', action='store_true', default=False, help='use the subset of datasets')
parser.add_argument('--log_epoch', default=5, type=int, help='print loss and metrix')
parser.add_argument('--draw_similar_matrix', default=10, type=int, help='draw similarity matrix')
parser.add_argument('--pretrain_lr', default=0.0001, type=float, help='pretrain learning rate')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--use_pretrain_epoch_dir', default=None, type=str,
                    help='choose the pretrain checkpoint to finetune')
parser.add_argument('--pretrain_epoch', default=10, type=int, help='pretrain epochs')
parser.add_argument('--finetune_epoch', default=300, type=int, help='finetune epochs')

parser.add_argument('--masking_ratio', default=0.5, type=float, help='masking ratio')
parser.add_argument('--positive_nums', default=3, type=int, help='positive series numbers')
parser.add_argument('--lm', default=3, type=int, help='average masked lenght')

parser.add_argument('--finetune_result_file_name', default="finetune_result.json", type=str,
                    help='finetune result json name')
parser.add_argument('--temperature', type=float, default=0.2, help='temperature')


def build_model(args, configs, device, ckpt_path):
    # Model Backbone
    model = TFC(configs, args).to(device)
    checkpoint = torch.load(ckpt_path)

    pretrained_dict = checkpoint["model_state_dict"]
    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model


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

from scipy.linalg import eigh

def stable_sqrtm_sym(matrix, eps=1e-10):
    """Efficient and stable square root of a symmetric positive-definite matrix using eigendecomposition."""
    vals, vecs = eigh(matrix)
    vals = np.clip(vals, eps, None)  # Ensure positive values
    sqrt_matrix = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
    return sqrt_matrix

def calculate_fid_optimized(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Efficient FID calculation for large covariance matrices using eigendecomposition-based square root."""
    diff = mu1 - mu2

    # Add small value to the diagonal to ensure numerical stability
    sigma1 += np.eye(sigma1.shape[0]) * eps
    sigma2 += np.eye(sigma2.shape[0]) * eps

    # Compute square root of the covariance product using eigendecomposition
    product = sigma1.dot(sigma2)
    covmean = stable_sqrtm_sym(product, eps)

    # Numerical stability check for imaginary values (which can occur due to computation issues)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Calculate FID score
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


def calculate_tempral_MSE(signal1, signal2):
    """Calculates the temporal MSE between two signals."""
    if signal1.shape != signal2.shape:
        raise ValueError("Input signals must have the same shape.")
    mse = np.mean((signal1 - signal2) ** 2)
    return mse

from scipy.signal import stft
from scipy import stats

# sample_freq = 250
# clip_fs = 40
# nperseg = 200
# noverlap = 190
def get_stft_stanford(x, fs=250, clip_fs=40, normalizing=None, **kwargs):
    f, t, Zxx = stft(x, fs, **kwargs)
    Zxx = Zxx[:, :clip_fs, :]
    f = f[:clip_fs]
    Zxx = np.abs(Zxx)
    clip = 5  # To handle boundary effects
    if normalizing == "zscore":
        Zxx = Zxx[:, :, clip:-clip]
        Zxx = stats.zscore(Zxx, axis=-1)
        t = t[clip:-clip]
        Zxx = np.nan_to_num(Zxx, nan=0)
    if np.isnan(Zxx).any():
        import pdb
        pdb.set_trace()
    return f, t, Zxx

def calculate_MSE(signal1, signal2, temp_mse):
    """Calculates the STFT MSE between two signals."""
    if signal1.shape != signal2.shape:
        raise ValueError("Input signals must have the same shape.")
    _,_,stft1 = get_stft_stanford(signal1, clip_fs=40,
                                  nperseg=200,noverlap=190,
                                  normalizing="zscore", return_onesided=True)
    _,_,stft2 = get_stft_stanford(signal2, clip_fs=40,
                                  nperseg=200,noverlap=190,
                                  normalizing="zscore", return_onesided=True)
    mse = np.mean((stft1 - stft2) ** 2)
    # final_mse = 0.5 * (temp_mse + mse)
    return mse#final_mse


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
    if data.ndim != 2:
        data = data.reshape(-1, data.shape[-1])
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


def line_length(x):
    """
    x: torch.Tensor or np.ndarray of shape (N, D)
    fs: sampling frequency (Hz)
    gamma_band: tuple defining high gamma band (Hz)

    Returns:
        features: torch.Tensor of shape (N, 4)
    """
    if x.ndim != 2:
        x = x.reshape(-1, x.shape[-1])

    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x

    # 1. Line Length
    line_length = np.sum(np.abs(np.diff(x_np, axis=1)), axis=1)
    line_length = np.nan_to_num(line_length, nan=0.0)
    line_length = np.expand_dims(line_length, axis=1)
    return line_length

from scipy.signal import butter, filtfilt
    # 4. High Gamma Power (using Butterworth bandpass filter)
def bandpass_filter_batch(data, low_freq, high_freq, fs, order=4):
    """
    Applies a Butterworth bandpass filter to the input data.
    :param data: Input data to be filtered.
    :param low_freq: Low cutoff frequency.
    :param high_freq: High cutoff frequency.
    :param fs: Sampling frequency.
    :param order: Order of the Butterworth filter.
    :return: Filtered data.
    """

    nyq = 0.5 * fs
    low = low_freq / nyq
    high = high_freq / nyq
    b, a = butter(order, [low, high], btype='band')
    # 使用 axis=-1 指定对时间轴滤波
    return filtfilt(b, a, data, axis=-1)

def gamma_highgamma(signal,fs=250, low_freq=30, high_freq=100, order=4):
    """
    Computes the gamma and high gamma energy of a signal.

    Parameters
    ----------
    signal : np.ndarray
        Input signal of shape (N, T), where N is the number of samples and T is the length of each sample.

    Returns
    -------
    np.ndarray
        Feature matrix of shape (N, 2), where each row contains [gamma_energy, high_gamma_energy].
    """

    # 批量滤波
    if signal.ndim != 2:
        X = signal.reshape(-1, signal.shape[-1])
    else:
        X = signal
    # 批量滤波
    X_gamma = bandpass_filter_batch(X, 30, 60, fs)
    X_high_gamma = bandpass_filter_batch(X, 60, 100, fs)

    # 批量能量计算（平均平方值）
    gamma_energy = np.mean(X_gamma ** 2, axis=1)         # shape [N]
    high_gamma_energy = np.mean(X_high_gamma ** 2, axis=1)  # shape [N]

    # 拼接为最终特征矩阵 [N, 2]
    features = np.stack([gamma_energy, high_gamma_energy], axis=1)
    return features

def count_spikes(data, threshold_factor=3.0):
    """
    data: [N, 1000] numpy array
    threshold_factor: spike 阈值 = mean + threshold_factor * std
    return: [N] spike 数组
    """
    if data.ndim != 2:
        data = data.reshape(-1, data.shape[-1])
    N, T = data.shape
    spike_counts = np.zeros(N, dtype=int)

    for i in range(N):
        signal = data[i]
        mean = np.mean(signal)
        std = np.std(signal)
        threshold = mean + threshold_factor * std

        # 一阶差分检测突变点（optional）
        diff_signal = np.abs(np.diff(signal))

        # 简单阈值检测（超过上阈值或下阈值）
        spike_locs = np.where(np.abs(signal - mean) > threshold)[0]

        # 可选去重：避免连续多个点被算作多个 spike
        if len(spike_locs) > 0:
            clean_spikes = [spike_locs[0]]
            for j in range(1, len(spike_locs)):
                if spike_locs[j] - clean_spikes[-1] > 10:  # 至少间隔 10 个采样点
                    clean_spikes.append(spike_locs[j])
            spike_counts[i] = len(clean_spikes)

    return np.expand_dims(spike_counts, axis=1)

# Set the parameters
sample_freq = 250

## Load the model ########################################################
# print('GPU available:', torch.cuda.is_available())
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     print('Using GPU')
# else:
device = torch.device("cpu")
print('Using CPU')

path = "/home/yuhxie/SimMTM/SimMTM_Classification/"
ckpt_path = path + "code/checkpoints/classification/SleepEEG_2_Epilepsy/ckpt_best.pt"

args, unknown = parser.parse_known_args()
exec(f'from config_files.{args.pretrain_dataset}_Configs import Config as Configs')
configs = Configs()
model = build_model(args, configs, device, ckpt_path)
model.to(device)

# Evaluate FND ##############################################################################
model.eval()  # Set model to evaluation mode
val_data_folder = "/net/inltitan1/scratch2/rpeng/SyntheticData/ValData/"
test_data_folder = "/net/inltitan1/scratch2/rpeng/SyntheticData/TestData/"
train_data_folder = "/net/inltitan1/scratch2/rpeng/SyntheticData/TrainData/"

UNet_data_folder = "/net/inltitan1/scratch2/rpeng/SyntheticData/UNet/"
DiT_data_folder = "/net/inltitan1/scratch2/rpeng/SyntheticData/DiT/"
CycleGAN_data_folder = "/net/inltitan1/scratch2/rpeng/SyntheticData/CycleGAN/"
# data_folder = "/home/rpeng/RPeng_workspace/DDPM/BCICIV_2a_gdf/"
# path_clean = val_data_folder + "ValData.npy"
path_clean = test_data_folder + "TestData.npy"



# # resample
# wav_clean = resample(wav_clean, 2000, axis=2)


# inputs_clean = torch.FloatTensor(wav_clean).to(device)
# with torch.no_grad():
#     out_clean1, out_clean2, out_clean = model.forward(inputs_clean)
#     out_clean1 = out_clean1.cpu().numpy()
#     out_clean2 = out_clean2.cpu().numpy()
#     out_clean = out_clean.cpu().numpy()

paths = [
UNet_data_folder + "UNet_generated_data.npy",
DiT_data_folder + "DiT_generated_data.npy",
CycleGAN_data_folder + "CycleGAN_generated_data.npy",
train_data_folder + "TrainData.npy"
]
for path_artifact in paths:
    # path_artifact = UNet_data_folder + "UNet_generated_data.npy"
    # path_artifact = DiT_data_folder + "DiT_generated_data.npy"
    # path_artifact = CycleGAN_data_folder + "CycleGAN_generated_data.npy"
    # path_artifact = train_data_folder + "TrainData.npy"

    wav_clean = np.load(path_clean, allow_pickle=True)  # .item()#['rest'].squeeze()
    print(wav_clean.shape)

    print(path_artifact)
    wav_artifact = np.load(path_artifact, allow_pickle=True)
    print(wav_artifact.shape)

    length1 = wav_clean.shape[0]
    length2 = wav_artifact.shape[0]
    sampler_size = 400 #min(length1, length2)
    #对两个数据集进行采样
    sample_idx1 = np.random.choice(length1, size=sampler_size, replace=False)
    sample_idx2 = np.random.choice(length2, size=sampler_size, replace=False)
    wav_clean = wav_clean[sample_idx1,:, :]
    wav_artifact = wav_artifact[sample_idx2,:,:]


    wav_clean = wav_clean.reshape(-1, wav_clean.shape[2])
    # wav_clean = np.expand_dims(wav_clean, axis=1)
    print(wav_clean.shape)

    # # resample
    # wav_clean = resample(wav_clean, 2000, axis=2)
    wav_artifact = wav_artifact.reshape(-1, wav_artifact.shape[2])
    # wav_artifact = np.expand_dims(wav_artifact, axis=1)
    print(wav_artifact.shape)
    # # resample
    # wav_artifact = resample(wav_artifact, 2000, axis=2)
    # wav_artifact = wav_artifact.reshape(-1, wav_artifact.shape[2])


    #特征提取
    # wav_clean_hjorth = hjorth_features_ndarray(wav_clean)
    # wav_artifact_hjorth = hjorth_features_ndarray(wav_artifact)
    #
    # wav_clean_linelength = line_length(wav_clean)
    # wav_artifact_linelength = line_length(wav_artifact)
    #
    # wav_clean_gamma = gamma_highgamma(wav_clean)
    # wav_artifact_gamma = gamma_highgamma(wav_artifact)
    #
    # wav_clean_spike = count_spikes(wav_clean)
    # wav_artifact_spike = count_spikes(wav_artifact)

    # _, _, wav_clean_stft = get_stft_stanford(wav_clean, clip_fs=40,
    #                                 nperseg=200, noverlap=190,
    #                                 normalizing="zscore", return_onesided=True)
    # wav_clean_stft = wav_clean_stft.reshape(wav_clean_stft.shape[0],-1)
    # _, _, wav_artifact_stft = get_stft_stanford(wav_artifact, clip_fs=40,
    #                                 nperseg=200, noverlap=190,
    #                                 normalizing="zscore", return_onesided=True)
    # wav_artifact_stft = wav_artifact_stft.reshape(wav_artifact_stft.shape[0],-1)
    #
    # full_features_clean = np.concatenate((wav_clean_hjorth, wav_clean_linelength,wav_clean_gamma, wav_clean_spike,wav_clean_stft), axis=1)
    # full_features_artifact = np.concatenate((wav_artifact_hjorth, wav_artifact_linelength,wav_artifact_gamma, wav_artifact_spike,wav_artifact_stft), axis=1)
    # #
    # print(wav_clean.shape)#N,1,1000
    # print(wav_artifact.shape)#N,1,1000
    # print(full_features_clean.shape)#N,1,1000
    # print(full_features_artifact.shape)#N,1,1000

    # hjorth_sme = calculate_tempral_MSE(wav_clean_hjorth, wav_artifact_hjorth)
    # print(f"Handcrafted Features Hjorth: {hjorth_sme}")
    #
    # linelength_sme = calculate_tempral_MSE(wav_clean_linelength, wav_artifact_linelength)
    # print(f"Handcrafted Features LineLength: {linelength_sme}")
    #
    # gamma_sme = calculate_tempral_MSE(wav_clean_gamma, wav_artifact_gamma)
    # print(f"Handcrafted Features GAMMA: {gamma_sme}")
    #
    # spike_sme = calculate_tempral_MSE(wav_clean_spike, wav_artifact_spike)
    # print(f"Handcrafted Features SPIKE: {spike_sme}")
    #
    # mymse = calculate_MSE(wav_clean, wav_artifact, 0)
    # print(f"Handcrafted Features STFT: {mymse}")

    # feature_sme = calculate_tempral_MSE(full_features_clean, full_features_artifact)
    # print(f"Handcrafted Features Full: {feature_sme}")

    gt_sme = calculate_tempral_MSE(wav_clean, wav_artifact)
    print(f"Handcrafted Features GT: {gt_sme}")

    mu_clean = np.mean(wav_clean, axis=0)
    sigma_clean = np.cov(wav_clean, rowvar=False)

    mu_artifact = np.mean(wav_artifact, axis=0)
    sigma_artifact = np.cov(wav_artifact, rowvar=False)

    # Compute the Fréchet Neural Distance
    fid = calculate_fid(mu_clean, sigma_clean, mu_artifact, sigma_artifact)
    print(f"Fréchet Neural Distance (FND): {fid}")




    # inputs_artifact = torch.FloatTensor(wav_artifact).to(device)
    # with torch.no_grad():
    #     out_artifact1, out_artifact2, out_artifact = model.forward(inputs_artifact)
    #     out_artifact1 = out_artifact1.cpu().numpy()
    #     out_artifact2 = out_artifact2.cpu().numpy()
    #     out_artifact = out_artifact.cpu().numpy()

    ### FND
    # Flatten last two dimensions (combine the 40 and 768)

    #
    # test_last_layer = False
    # if test_last_layer:
    #     print("test last layer")
    #     # out_clean = out_clean.mean(dim=1)
    #     # out_artifact = out_artifact.mean(dim=1)
    #     out_clean_flattened = out_clean.reshape(out_clean.shape[0], -1)
    #     out_artifact_flattened = out_artifact.reshape(out_artifact.shape[0], -1)
    # else:
    #     print("test 2nd layer")
    #     print(out_clean2.shape)
    #     # out_clean2 = np.mean(out_clean2, axis=1)
    #     # out_artifact2 = np.mean(out_artifact2, axis=1)
    #     out_clean_flattened = out_clean2.reshape(out_clean2.shape[0], -1)
    #     out_artifact_flattened = out_artifact2.reshape(out_artifact2.shape[0], -1)
    #
    # print(out_clean_flattened.shape)
    #
    # # visualize_tsne(out_clean_flattened, out_artifact_flattened, label1 = 'seizure', label2 = 'nonseizure', layer = 'first' if test_first_layer else 'last')
    #
    # # Calculate the means and covariances of the encoded datasets
    # mu_clean = np.mean(out_clean_flattened, axis=0)
    # sigma_clean = np.cov(out_clean_flattened, rowvar=False)
    #
    # mu_artifact = np.mean(out_artifact_flattened, axis=0)
    # sigma_artifact = np.cov(out_artifact_flattened, rowvar=False)
    #
    # # Compute the Fréchet Neural Distance
    # fid = calculate_fid(mu_clean, sigma_clean, mu_artifact, sigma_artifact)
    # print(f"Fréchet Neural Distance (FND): {fid}")
    #
    # mu_clean = np.mean(full_features_clean, axis=0)
    # sigma_clean = np.cov(full_features_clean, rowvar=False)
    #
    # mu_artifact = np.mean(full_features_artifact, axis=0)
    # sigma_artifact = np.cov(full_features_artifact, rowvar=False)
    #
    # # Compute the Fréchet Neural Distance
    # fid = calculate_fid(mu_clean, sigma_clean, mu_artifact, sigma_artifact)
    # print(f"Fréchet Neural Distance (FND): {fid}")