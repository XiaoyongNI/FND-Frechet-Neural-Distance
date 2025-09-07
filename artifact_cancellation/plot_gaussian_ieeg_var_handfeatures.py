import numpy as np
import matplotlib.pyplot as plt

data_folder = '/net/inltitan1/scratch2/yuhxie/BrainBERT/'
raw1 = np.load(data_folder+'Gaussian_check_results_non_seizure_raw_N50_5k.npy', allow_pickle=True).item()
raw2 = np.load(data_folder+'Gaussian_check_results_non_seizure_raw_N10k_100k.npy',  allow_pickle=True).item()

stft1 = np.load(data_folder+'Gaussian_check_results_non_seizure_stft_N50_5k.npy', allow_pickle=True).item()
stft2 = np.load(data_folder+'Gaussian_check_results_non_seizure_stft_N10k_100k.npy', allow_pickle=True).item()

data_folder = 'FD_ValidationCheck/'
Hjorth1 = np.load(data_folder+'Gaussian_check_results_Hjorth_non_seizure_N50_5k.npy', allow_pickle=True).item()
Hjorth2 = np.load(data_folder+'Gaussian_check_results_Hjorth_non_seizure_N10k_100k.npy', allow_pickle=True).item()

LL1 = np.load(data_folder+'Gaussian_check_results_LL_non_seizure_N50_5k.npy', allow_pickle=True).item()
LL2 = np.load(data_folder+'Gaussian_check_results_LL_non_seizure_N10k_100k.npy', allow_pickle=True).item()

SK1 = np.load(data_folder+'Gaussian_check_results_SK_non_seizure_N50_5k.npy', allow_pickle=True).item()
SK2 = np.load(data_folder+'Gaussian_check_results_SK_non_seizure_N10k_100k.npy', allow_pickle=True).item()

Kur1 = np.load(data_folder+'Gaussian_check_results_Kur_non_seizure_N50_5k.npy', allow_pickle=True).item()
Kur2 = np.load(data_folder+'Gaussian_check_results_Kur_non_seizure_N10k_100k.npy', allow_pickle=True).item()

BP1 = np.load(data_folder+'Gaussian_check_results_BP_non_seizure_N50_5k.npy', allow_pickle=True).item()
BP2 = np.load(data_folder+'Gaussian_check_results_BP_non_seizure_N10k_100k.npy', allow_pickle=True).item()

all1 = np.load(data_folder+'Gaussian_check_results_all_non_seizure_N50_5k.npy', allow_pickle=True).item()
all2 = np.load(data_folder+'Gaussian_check_results_all_non_seizure_N10k_100k.npy', allow_pickle=True).item()

N = [50,100,200,500,1000,5000,10000, 20000, 50000, 100000]

raw = []
stft = []
Hjorth = []
LL = []
SK = []
Kur = []
BP = []
all = []
cond = "non_seizure"

for ks_tuple in raw1[cond]['var']:
    raw.append(ks_tuple[0])

for ks_tuple in stft1[cond]['var']:
    stft.append(ks_tuple[0])
    
for ks_tuple in Hjorth1[cond]['var']:
    Hjorth.append(ks_tuple[0])

for ks_tuple in LL1[cond]['var']:
    LL.append(ks_tuple[0])

for ks_tuple in SK1[cond]['var']:
    SK.append(ks_tuple[0])
    
for ks_tuple in Kur1[cond]['var']:
    Kur.append(ks_tuple[0])
    
for ks_tuple in BP1[cond]['var']:
    BP.append(ks_tuple[0])
    
for ks_tuple in all1[cond]['var']:
    all.append(ks_tuple[0])


### N = 10k - 100k
for ks_tuple in raw2[cond]['var']:
    raw.append(ks_tuple[0])

for ks_tuple in stft2[cond]['var']:
    stft.append(ks_tuple[0])
    
for ks_tuple in Hjorth2[cond]['var']:
    Hjorth.append(ks_tuple[0])
    
for ks_tuple in LL2[cond]['var']:
    LL.append(ks_tuple[0])
    
for ks_tuple in SK2[cond]['var']:
    SK.append(ks_tuple[0])
    
for ks_tuple in Kur2[cond]['var']:
    Kur.append(ks_tuple[0])
    
for ks_tuple in BP2[cond]['var']:
    BP.append(ks_tuple[0])
    
for ks_tuple in all2[cond]['var']:
    all.append(ks_tuple[0])
    

colors = {
    # 'Raw': '#FFD700',        # Bright Yellow (Gold)
    'individual': '#FFA500',       # Orange
    'all': '#8B4513',     # SaddleBrown (Dark Brownish Orange)

    'Threshold': 'red'               # KS reference line
}

# Draw the eeg raw stft lnm with x as cond, in same figure
plt.figure(figsize=(10, 5))
plt.plot(N, LL[0:len(N)], label=r'Line length$||\Sigma||$', marker='s', markersize=10, color=colors['individual'])
plt.plot(N, SK[0:len(N)], label=r'Skewness$||\Sigma||$', marker='o', markersize=10, color=colors['individual'])
plt.plot(N, Kur[0:len(N)], label=r'Kurtosis$||\Sigma||$', marker='v', markersize=10, color=colors['individual'])
plt.plot(N, BP[0:len(N)], label=r'HFO$||\Sigma||$', marker='d', markersize=10, color=colors['individual'])
plt.plot(N, stft[0:len(N)], label=r'STFT$||\Sigma||$', marker='^', markersize=10, color=colors['individual'])

plt.plot(N, all[0:len(N)], label=r'All$||\Sigma||$', marker='o', linewidth=3.5, color=colors['all'])


# # Log scale
plt.yscale('log')
# plt.xticks(N, rotation=45)
plt.xlabel('N', fontsize=20)    
plt.ylabel(r'$||\Sigma||$', rotation=90, fontsize=21) 
plt.ylim(top=plt.ylim()[1] * 1000000)  # Increase y-axis upper limit 
plt.tick_params(axis='both', which='major', labelsize=21) 
plt.legend(loc='upper center', ncol=3, fontsize=19)
plt.grid()
plt.tight_layout()
plt.savefig('Var_check_ieeg_results_handfeatures.png')
