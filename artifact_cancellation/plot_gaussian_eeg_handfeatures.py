import numpy as np
import matplotlib.pyplot as plt


data_folder = 'FD_ValidationCheck/'
Hjorth1 = np.load(data_folder+'Gaussian_check_eeg_results_Hjorth_non_seizure_N50_5k.npy', allow_pickle=True).item()

Spike1 = np.load(data_folder+'Gaussian_check_eeg_results_Spike_non_seizure_N50_5k.npy', allow_pickle=True).item()

LL1 = np.load(data_folder+'Gaussian_check_eeg_results_LL_non_seizure_N50_5k.npy', allow_pickle=True).item()

BP1 = np.load(data_folder+'Gaussian_check_eeg_results_BP_non_seizure_N50_5k.npy', allow_pickle=True).item()

STFT1 = np.load(data_folder+'Gaussian_check_eeg_results_STFT_non_seizure_N50_5k.npy', allow_pickle=True).item()

all1 = np.load(data_folder+'Gaussian_check_eeg_results_all_non_seizure_N50_5k.npy', allow_pickle=True).item()

N = [50,100,200,500]#,1000] #5000,10000, 20000, 50000, 100000


Hjorth = []
Spike = []
LL = []
BP = []
STFT = []
all = []

cond = "non_seizure"


    
for ks_tuple in Hjorth1[cond]['ks']:
    Hjorth.append(ks_tuple[0])
    
for ks_tuple in Spike1[cond]['ks']:
    Spike.append(ks_tuple[0])

for ks_tuple in LL1[cond]['ks']:
    LL.append(ks_tuple[0])
    
for ks_tuple in BP1[cond]['ks']:
    BP.append(ks_tuple[0])
    
for ks_tuple in STFT1[cond]['ks']:
    STFT.append(ks_tuple[0])
    
for ks_tuple in all1[cond]['ks']:
    all.append(ks_tuple[0]) 
    
### N = 10k - 100k

    

colors = {
    'individual': '#CBAACB',  # Pastel light purple (custom, pleasant tone)
    'all': '#4B0082',  # Indigo (deep, dark purple)
    'Threshold': 'red'               # KS reference line
}

# Draw the eeg raw stft lnm with x as cond, in same figure
plt.figure(figsize=(10, 5))
plt.plot(N, Hjorth[0:len(N)], label='Hjorth', marker='o', markersize=10, color=colors['individual'])
plt.plot(N, Spike[0:len(N)], label='Spike', marker='v', markersize=10, color=colors['individual'])
plt.plot(N, LL[0:len(N)], label='Line length', marker='s', markersize=10, color=colors['individual'])
plt.plot(N, BP[0:len(N)], label='Band power', marker='d', markersize=10, color=colors['individual'])
plt.plot(N, STFT[0:len(N)], label='STFT', marker='^', markersize=10, color=colors['individual'])

plt.plot(N, all[0:len(N)], label='All', marker='o', linewidth=3.5, color=colors['all'])

# Add D_critical = 1.36/sqrt(N)
D_critical = 1.36 / np.sqrt(N)
plt.plot(N, D_critical, color=colors['Threshold'], linestyle='--', linewidth=3.5,
         label=r'$D_{\mathrm{critical}}$') # = \frac{1.36}{\sqrt{N}}

# # Log scale for x-axis
# plt.xscale('log')
# plt.xticks(N, rotation=45)
plt.xlabel('N', fontsize=20)    
plt.ylabel(r'$D_N$', fontsize=20) 
plt.ylim(top=plt.ylim()[1] * 1.7)  # Increase y-axis upper limit by 15% 
plt.tick_params(axis='both', which='major', labelsize=20) 
plt.legend(loc='upper center', ncol=3, fontsize=21)
plt.grid()
plt.tight_layout()
plt.savefig('Gaussian_check_eeg_results_handfeatures.png')
