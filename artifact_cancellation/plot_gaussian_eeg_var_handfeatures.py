import numpy as np
import matplotlib.pyplot as plt

data_folder = 'FD_ValidationCheck/'
Hjorth1 = np.load(data_folder+'Gaussian_check_eeg_results_Hjorth_non_seizure_N50_5k.npy', allow_pickle=True).item()
Hjorth2 = np.load(data_folder+'Gaussian_check_eeg_results_Hjorth_non_seizure_N10k_100k.npy', allow_pickle=True).item()

Spike1 = np.load(data_folder+'Gaussian_check_eeg_results_Spike_non_seizure_N50_5k.npy', allow_pickle=True).item()
Spike2 = np.load(data_folder+'Gaussian_check_eeg_results_Spike_non_seizure_N10k_100k.npy', allow_pickle=True).item()

LL1 = np.load(data_folder+'Gaussian_check_eeg_results_LL_non_seizure_N50_5k.npy', allow_pickle=True).item()
LL2 = np.load(data_folder+'Gaussian_check_eeg_results_LL_non_seizure_N10k_100k.npy', allow_pickle=True).item()

BP1 = np.load(data_folder+'Gaussian_check_eeg_results_BP_non_seizure_N50_5k.npy', allow_pickle=True).item()
BP2 = np.load(data_folder+'Gaussian_check_eeg_results_BP_non_seizure_N10k_100k.npy', allow_pickle=True).item()

STFT1 = np.load(data_folder+'Gaussian_check_eeg_results_STFT_non_seizure_N50_5k.npy', allow_pickle=True).item()
STFT2 = np.load(data_folder+'Gaussian_check_eeg_results_STFT_non_seizure_N10k_100k.npy', allow_pickle=True).item()

all1 = np.load(data_folder+'Gaussian_check_eeg_results_all_non_seizure_N50_5k.npy', allow_pickle=True).item()
all2 = np.load(data_folder+'Gaussian_check_eeg_results_all_non_seizure_N10k_100k.npy', allow_pickle=True).item()

N = [50,100,200,500,1000, 5000,10000, 20000, 50000, 100000]

Hjorth = []
Spike = []
LL = []
BP = []
STFT = []
all = []
cond = "non_seizure"

for ks_tuple in Hjorth1[cond]['var']:
    Hjorth.append(ks_tuple[0])
    
for ks_tuple in Spike1[cond]['var']:
    Spike.append(ks_tuple[0])

for ks_tuple in LL1[cond]['var']:
    LL.append(ks_tuple[0])
    
for ks_tuple in BP1[cond]['var']:
    BP.append(ks_tuple[0])
    
for ks_tuple in STFT1[cond]['var']:
    STFT.append(ks_tuple[0])
    
for ks_tuple in all1[cond]['var']:
    all.append(ks_tuple[0]) 
    
### N = 10k - 100k
for ks_tuple in Hjorth2[cond]['var']:
    Hjorth.append(ks_tuple[0])
    
for ks_tuple in Spike2[cond]['var']:
    Spike.append(ks_tuple[0])
    
for ks_tuple in LL2[cond]['var']:
    LL.append(ks_tuple[0])
    
for ks_tuple in BP2[cond]['var']:
    BP.append(ks_tuple[0])
    
for ks_tuple in STFT2[cond]['var']:
    STFT.append(ks_tuple[0])
    
for ks_tuple in all2[cond]['var']:
    all.append(ks_tuple[0])
    
    
colors = {
    'individual': '#CBAACB',  # Pastel light purple (custom, pleasant tone)
    'all': '#4B0082',  # Indigo (deep, dark purple)
    'Threshold': 'red'               # KS reference line
}


# Draw the eeg raw stft lnm with x as cond, in same figure
plt.figure(figsize=(10, 5))
plt.plot(N, Hjorth[0:len(N)], label=r'Hjorth$||\Sigma||$', marker='o', markersize=10, color=colors['individual'])
plt.plot(N, Spike[0:len(N)], label=r'Spike$||\Sigma||$', marker='v', markersize=10, color=colors['individual'])
plt.plot(N, LL[0:len(N)], label=r'Line length$||\Sigma||$', marker='s', markersize=10, color=colors['individual'])
plt.plot(N, BP[0:len(N)], label=r'Band power$||\Sigma||$', marker='d', markersize=10, color=colors['individual'])
plt.plot(N, STFT[0:len(N)], label=r'STFT$||\Sigma||$', marker='^', markersize=10, color=colors['individual'])

plt.plot(N, all[0:len(N)], label=r'All$||\Sigma||$', marker='o', linewidth=3.5, color=colors['all'])
#Log
plt.yscale('log')
# plt.xticks(N, rotation=45)
plt.xlabel('N', fontsize=20)    
plt.ylabel(r'$||\Sigma||$', rotation=90, fontsize=21) 
plt.ylim(top=plt.ylim()[1] * 10000)  # Increase y-axis upper limit 
plt.tick_params(axis='both', which='major', labelsize=21) 
plt.legend(loc='upper center', ncol=3, fontsize=19)
plt.grid()
plt.tight_layout()
plt.savefig('Var_check_eeg_results_handfeatures.png')
