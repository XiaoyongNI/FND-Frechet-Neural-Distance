import numpy as np
import matplotlib.pyplot as plt

data_folder = '/home/yuhxie/BrainBERT/'
eeg1 = np.load(data_folder+'Gaussian_check_results_non_seizure_EEGNet_N50_5k.npy', allow_pickle=True).item()
eeg2 = np.load(data_folder+'Gaussian_check_results_non_seizure_EEGNet_N10k_100k.npy', allow_pickle=True).item()

eegmulti1 = np.load(data_folder+'Gaussian_check_results_non_seizure_eegmulti1_N50_5k.npy', allow_pickle=True).item()
eegmulti2 = np.load(data_folder+'Gaussian_check_results_non_seizure_eegmulti1_N10k_100k.npy', allow_pickle=True).item()

raw1 = np.load(data_folder+'Gaussian_check_results_non_seizure_raw_N50_5k.npy', allow_pickle=True).item()
raw2 = np.load(data_folder+'Gaussian_check_results_non_seizure_raw_N10k_100k.npy',  allow_pickle=True).item()

stft1 = np.load(data_folder+'Gaussian_check_results_non_seizure_stft_N50_5k.npy', allow_pickle=True).item()
stft2 = np.load(data_folder+'Gaussian_check_results_non_seizure_stft_N10k_100k.npy', allow_pickle=True).item()

Hjorth1 = np.load('/net/inltitan1/scratch2/Xiaoyong/Artifact_cancellation/Gaussian_check_results_non_seizure_Hjorth_N50_5k.npy', allow_pickle=True).item()
Hjorth2 = np.load('/net/inltitan1/scratch2/Xiaoyong/Artifact_cancellation/Gaussian_check_results_non_seizure_Hjorth_N10k_100k.npy', allow_pickle=True).item()

lnm1 = np.load('/net/inltitan1/scratch2/Xiaoyong/Artifact_cancellation/Gaussian_check_results_non_seizure_N50_5k.npy', allow_pickle=True).item()
lnm2 = np.load('/net/inltitan1/scratch2/Xiaoyong/Artifact_cancellation/Gaussian_check_results_non_seizure_N10k_100k.npy', allow_pickle=True).item()

firstlayer1 = np.load(data_folder+'Gaussian_check_results_non_seizure_brainbert1_N50_5k.npy', allow_pickle=True).item()
firstlayer2 = np.load(data_folder+'Gaussian_check_results_non_seizure_brainbert1_N10k_100k.npy', allow_pickle=True).item()

N = [50,100,200,500,1000,5000,10000, 20000, 50000, 100000]

eeg = []
eegmulti = []
raw = []
stft = []
Hjorth = []
lnm = []
firstlayer = []

cond = "non_seizure"

for ks_tuple in eeg1[cond]['var']:
    eeg.append(ks_tuple[0])
    
for ks_tuple in eegmulti1[cond]['var']:
    eegmulti.append(ks_tuple[0])

for ks_tuple in raw1[cond]['var']:
    raw.append(ks_tuple[0])

for ks_tuple in stft1[cond]['var']:
    stft.append(ks_tuple[0])
    
for ks_tuple in Hjorth1[cond]['var']:
    Hjorth.append(ks_tuple[0])

for ks_tuple in lnm1[cond]['var']:
    lnm.append(ks_tuple[0])

for ks_tuple in firstlayer1[cond]['var']:
    firstlayer.append(ks_tuple[0])

for ks_tuple in eeg2[cond]['var']:
    eeg.append(ks_tuple[0])
    
for ks_tuple in eegmulti2[cond]['var']:
    eegmulti.append(ks_tuple[0])

for ks_tuple in raw2[cond]['var']:
    raw.append(ks_tuple[0])

for ks_tuple in stft2[cond]['var']:
    stft.append(ks_tuple[0])
    
for ks_tuple in Hjorth2[cond]['var']:
    Hjorth.append(ks_tuple[0])

for ks_tuple in lnm2[cond]['var']:
    lnm.append(ks_tuple[0])
    
for ks_tuple in firstlayer2[cond]['var']:
    firstlayer.append(ks_tuple[0])

lnm.append(lnm[-1])

print(len(eeg))
print(len(raw))
print(len(stft))
print(len(lnm))


colors = {
    'CNN-1pat': 'royalblue',         # base CNN single-patient
    'CNN-multipat': 'navy',          # same family, darker blue
    'Raw': '#FFD700',        # Bright Yellow (Gold)
    'STFT': '#FFA500',       # Orange
    'Hjorth': '#8B4513',     # SaddleBrown (Dark Brownish Orange)
    'BrainBERT-lastLayer': 'darkgreen',   # base model green
    'BrainBERT-1stLayer': 'mediumseagreen', # same family, darker green
    'Threshold': 'red'               # KS reference line
}

# Draw the eeg raw stft lnm with x as cond, in same figure
plt.figure(figsize=(10, 5))
plt.plot(N, eeg[0:len(N)], label=r'CNN-1 patient$||\Sigma||$', marker='o', linewidth=3, color=colors['CNN-1pat'])
plt.plot(N, eegmulti[0:len(N)], label=r'CNN-multipatients$||\Sigma||$', marker='o', linewidth=3, color=colors['CNN-multipat'])
plt.plot(N, lnm[0:len(N)], label=r'BrainBERT-lastLayer$||\Sigma||$', marker='o', linewidth=3, color=colors['BrainBERT-lastLayer'])
plt.plot(N, firstlayer[0:len(N)], label=r'BrainBERT-1stLayer$||\Sigma||$', marker='o', linewidth=3, color=colors['BrainBERT-1stLayer'])

# # Log scale
plt.yscale('log')
# plt.xticks(N, rotation=45)
plt.xlabel('N', fontsize=20)    
plt.ylabel(r'$||\Sigma||$', rotation=90, fontsize=21) 
plt.ylim(top=plt.ylim()[1] * 10)  # Increase y-axis upper limit 
plt.tick_params(axis='both', which='major', labelsize=21) 
plt.legend(loc='upper center', ncol=2, fontsize=19)
plt.grid()
plt.tight_layout()
plt.savefig('Var_check_ieeg_results_non_seizure.png')
