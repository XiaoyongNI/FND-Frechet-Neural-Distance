import numpy as np
import matplotlib.pyplot as plt

data_folder = '/net/inltitan1/scratch2/yuhxie/BrainBERT/'
raw1 = np.load(data_folder+'Gaussian_check_results_non_seizure_raw_N50_5k.npy', allow_pickle=True).item()
raw2 = np.load(data_folder+'Gaussian_check_results_non_seizure_raw_N10k_100k.npy',  allow_pickle=True).item()

stft1 = np.load(data_folder+'Gaussian_check_results_non_seizure_stft_N50_5k.npy', allow_pickle=True).item()
stft2 = np.load(data_folder+'Gaussian_check_results_non_seizure_stft_N10k_100k.npy', allow_pickle=True).item()

data_folder = 'FD_ValidationCheck/'
Hjorth1 = np.load(data_folder+'Gaussian_check_results_Hjorth_non_seizure_N50_5k.npy', allow_pickle=True).item()
# Hjorth2 = np.load('/net/inltitan1/scratch2/Xiaoyong/Artifact_cancellation/Gaussian_check_results_non_seizure_Hjorth_N10k_100k.npy', allow_pickle=True).item()

LL1 = np.load(data_folder+'Gaussian_check_results_LL_non_seizure_N50_5k.npy', allow_pickle=True).item()

SK1 = np.load(data_folder+'Gaussian_check_results_SK_non_seizure_N50_5k.npy', allow_pickle=True).item()

Kur1 = np.load(data_folder+'Gaussian_check_results_Kur_non_seizure_N50_5k.npy', allow_pickle=True).item()

BP1 = np.load(data_folder+'Gaussian_check_results_BP_non_seizure_N50_5k.npy', allow_pickle=True).item()

all1 = np.load(data_folder+'Gaussian_check_results_all_non_seizure_N50_5k.npy', allow_pickle=True).item()

N = [50,100,200,500]#,1000] #5000,10000, 20000, 50000, 100000


raw = []
stft = []
Hjorth = []
LL = []
SK = []
Kur = []
BP = []
all = []

cond = "non_seizure"


for ks_tuple in raw1[cond]['ks']:
    raw.append(ks_tuple[0])

for ks_tuple in stft1[cond]['ks']:
    stft.append(ks_tuple[0])
    
for ks_tuple in Hjorth1[cond]['ks']:
    Hjorth.append(ks_tuple[0])

for ks_tuple in LL1[cond]['ks']:
    LL.append(ks_tuple[0])

for ks_tuple in SK1[cond]['ks']:
    SK.append(ks_tuple[0])
    
for ks_tuple in Kur1[cond]['ks']:
    Kur.append(ks_tuple[0])
    
for ks_tuple in BP1[cond]['ks']:
    BP.append(ks_tuple[0])
    
for ks_tuple in all1[cond]['ks']:
    all.append(ks_tuple[0])


    
    
    
### N = 10k - 100k
# for ks_tuple in raw2[cond]['ks']:
#     raw.append(ks_tuple[0])

# for ks_tuple in stft2[cond]['ks']:
#     stft.append(ks_tuple[0])
    




colors = {
    # 'Raw': '#FFD700',        # Bright Yellow (Gold)
    'individual': '#FFA500',       # Orange
    'all': '#8B4513',     # SaddleBrown (Dark Brownish Orange)

    'Threshold': 'red'               # KS reference line
}

# Draw the eeg raw stft lnm with x as cond, in same figure
plt.figure(figsize=(10, 5))
# plt.plot(N, raw[0:len(N)], label='Raw signal', marker='x', markersize=10, color=colors['individual'])
# plt.plot(N, Hjorth[0:len(N)], label='Hjorth', marker='*', markersize=10, color=colors['individual'])
plt.plot(N, LL[0:len(N)], label='Line length', marker='s', markersize=10, color=colors['individual'])
plt.plot(N, SK[0:len(N)], label='Skewness', marker='o', markersize=10, color=colors['individual'])
plt.plot(N, Kur[0:len(N)], label='Kurtosis', marker='v', markersize=10, color=colors['individual'])
plt.plot(N, BP[0:len(N)], label='HFO', marker='d', markersize=10, color=colors['individual'])
plt.plot(N, stft[0:len(N)], label='STFT', marker='^', markersize=10, color=colors['individual'])

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
plt.savefig('Gaussian_check_ieeg_results_handfeatures.png')
