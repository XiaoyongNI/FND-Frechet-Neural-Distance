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

for ks_tuple in Hjorth1[cond]['mu']:
    Hjorth.append(ks_tuple[0])
    
for ks_tuple in Spike1[cond]['mu']:
    Spike.append(ks_tuple[0])

for ks_tuple in LL1[cond]['mu']:
    LL.append(ks_tuple[0])
    
for ks_tuple in BP1[cond]['mu']:
    BP.append(ks_tuple[0])
    
for ks_tuple in STFT1[cond]['mu']:
    STFT.append(ks_tuple[0])
    
for ks_tuple in all1[cond]['mu']:
    all.append(ks_tuple[0]) 
    
### N = 10k - 100k
for ks_tuple in Hjorth2[cond]['mu']:
    Hjorth.append(ks_tuple[0])
    
for ks_tuple in Spike2[cond]['mu']:
    Spike.append(ks_tuple[0])
    
for ks_tuple in LL2[cond]['mu']:
    LL.append(ks_tuple[0])
    
for ks_tuple in BP2[cond]['mu']:
    BP.append(ks_tuple[0])
    
for ks_tuple in STFT2[cond]['mu']:
    STFT.append(ks_tuple[0])
    
for ks_tuple in all2[cond]['mu']:
    all.append(ks_tuple[0])
    
    
colors = {
    'individual': '#CBAACB',  # Pastel light purple (custom, pleasant tone)
    'all': '#4B0082',  # Indigo (deep, dark purple)
    'Threshold': 'red'               # KS reference line
}


# Draw the eeg raw stft lnm with x as cond, in same figure
plt.figure(figsize=(10, 5))
plt.plot(N, Hjorth[0:len(N)], label=r'Hjorth$||\mu||$', marker='o', markersize=10, color=colors['individual'])
plt.plot(N, Spike[0:len(N)], label=r'Spike$||\mu||$', marker='v', markersize=10, color=colors['individual'])
plt.plot(N, LL[0:len(N)], label=r'Line length$||\mu||$', marker='s', markersize=10, color=colors['individual'])
plt.plot(N, BP[0:len(N)], label=r'Band power$||\mu||$', marker='d', markersize=10, color=colors['individual'])
plt.plot(N, STFT[0:len(N)], label=r'STFT$||\mu||$', marker='^', markersize=10, color=colors['individual'])

plt.plot(N, all[0:len(N)], label=r'All$||\mu||$', marker='o', linewidth=3.5, color=colors['all'])
#Log
plt.yscale('log')
# plt.xticks(N, rotation=45)
plt.xlabel('N', fontsize=20)    
plt.ylabel(r'$||\mu||$', rotation=90, fontsize=21) 
plt.ylim(top=plt.ylim()[1] * 100)  # Increase y-axis upper limit 
plt.tick_params(axis='both', which='major', labelsize=21) 
plt.legend(loc='upper center', ncol=3, fontsize=19)
plt.grid()
plt.tight_layout()
plt.savefig('Mu_check_eeg_results_handfeatures.png')
