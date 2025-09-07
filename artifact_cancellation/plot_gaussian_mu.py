import numpy as np
import matplotlib.pyplot as plt

eeg1 = np.load('Gaussian_check_results_non_seizure_EEGNet_N50_5k.npy', allow_pickle=True).item()
eeg2 = np.load('Gaussian_check_results_non_seizure_EEGNet_N10k_100k.npy', allow_pickle=True).item()

raw1 = np.load('Gaussian_check_results_non_seizure_raw_N50_5k.npy', allow_pickle=True).item()
raw2 = np.load('Gaussian_check_results_non_seizure_raw_N10k_100k.npy',  allow_pickle=True).item()

stft1 = np.load('Gaussian_check_results_non_seizure_stft_N50_5k.npy', allow_pickle=True).item()
stft2 = np.load('Gaussian_check_results_non_seizure_stft_N10k_100k.npy', allow_pickle=True).item()

lnm1 = np.load('/net/inltitan1/scratch2/Xiaoyong/Artifact_cancellation/Gaussian_check_results_non_seizure_N50_5k.npy', allow_pickle=True).item()
lnm2 = np.load('/net/inltitan1/scratch2/Xiaoyong/Artifact_cancellation/Gaussian_check_results_non_seizure_N10k_100k.npy', allow_pickle=True).item()

N = [50,100,200,500,1000,5000,10000, 20000, 50000, 100000]

N = [1 /x for x in N]

eeg = []
raw = []
stft = []
lnm = []

cond = "non_seizure"

for ks_tuple in eeg1[cond]['mu']:
    eeg.append(ks_tuple[0])

for ks_tuple in raw1[cond]['mu']:
    raw.append(ks_tuple[0])

for ks_tuple in stft1[cond]['mu']:
    stft.append(ks_tuple[0])

for ks_tuple in lnm1[cond]['mu']:
    lnm.append(ks_tuple[0])

for ks_tuple in eeg2[cond]['mu']:
    eeg.append(ks_tuple[0])

for ks_tuple in raw2[cond]['mu']:
    raw.append(ks_tuple[0])

for ks_tuple in stft2[cond]['mu']:
    stft.append(ks_tuple[0])

for ks_tuple in lnm2[cond]['mu']:
    lnm.append(ks_tuple[0])

lnm.append(lnm[-1])

print(len(eeg))
print(len(raw))
print(len(stft))
print(len(lnm))


# draw the eeg raw stft lnm with x as cond, in same figure
plt.figure(figsize=(10, 5))
plt.plot(N, eeg, label='EEGNet', marker='o')
plt.plot(N, raw, label='RaW', marker='o')
plt.plot(N, stft, label='STFT', marker='o')
plt.plot(N, lnm, label='BrainBERT', marker='o')
#Log
plt.xscale('log')
plt.xticks(N, rotation=45)
plt.xlabel('1/N')
plt.ylabel('KS test statistic')
plt.title('KS test statistic of different models')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('Mu_check_results_non_seizure.png')
