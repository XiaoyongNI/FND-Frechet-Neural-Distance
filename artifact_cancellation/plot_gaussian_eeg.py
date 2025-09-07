import numpy as np
import matplotlib.pyplot as plt

data_folder = '/net/inltitan1/scratch2/yuhxie/BrainBERT/'
eeg1 = np.load(data_folder+'Gaussian_check_eeg_results_non_seizure_eegnetsingle_N50_5k.npy', allow_pickle=True).item()
eeg2 = np.load(data_folder+'Gaussian_check_eeg_results_non_seizure_eegnetsingle_N10k_100k.npy', allow_pickle=True).item()

eegmulti1 = np.load(data_folder+'Gaussian_check_eeg_results_non_seizure_eegnetmulti_N50_5k.npy', allow_pickle=True).item()
eegmulti2 = np.load(data_folder+'Gaussian_check_eeg_results_non_seizure_eegnetmulti_N10k_100k.npy', allow_pickle=True).item()

raw1 = np.load(data_folder+'Gaussian_check_eeg_results_non_seizure_raw_N50_5k.npy', allow_pickle=True).item()
raw2 = np.load(data_folder+'Gaussian_check_eeg_results_non_seizure_raw_N10k_100k.npy',  allow_pickle=True).item()

stft1 = np.load(data_folder+'Gaussian_check_eeg_results_non_seizure_stft_N50_5k.npy', allow_pickle=True).item()
stft2 = np.load(data_folder+'Gaussian_check_eeg_results_non_seizure_stft_N10k_100k.npy', allow_pickle=True).item()


SimMTM_folder = "/net/inltitan1/scratch2/Xiaoyong/Artifact_cancellation/SimMTM/512Hz/"
Hjorth1 = np.load(SimMTM_folder+'Gaussian_check_eeg_results_non_seizure_Hjorth_N50_5k.npy', allow_pickle=True).item()
Hjorth2 = np.load(SimMTM_folder+'Gaussian_check_eeg_results_non_seizure_Hjorth_N10k_100k.npy', allow_pickle=True).item()

firstlayer1 = np.load(SimMTM_folder+'avg_SimMTM/Gaussian_check_eeg_results_non_seizure_SimMTM1stlayer_N50_5k.npy', allow_pickle=True).item()
firstlayer2 = np.load(SimMTM_folder+'avg_SimMTM/Gaussian_check_eeg_results_non_seizure_SimMTM1stlayer_N10k_100k.npy', allow_pickle=True).item()

secondlayer1 = np.load(SimMTM_folder+'avg_SimMTM/Gaussian_check_eeg_results_non_seizure_SimMTM2ndlayer_N50_5k.npy', allow_pickle=True).item()
secondlayer2 = np.load(SimMTM_folder+'avg_SimMTM/Gaussian_check_eeg_results_non_seizure_SimMTM2ndlayer_N10k_100k.npy', allow_pickle=True).item()

simMTM1 = np.load(SimMTM_folder+'avg_SimMTM/Gaussian_check_eeg_results_non_seizure_SimMTMfinallayer_N50_5k.npy', allow_pickle=True).item()
simMTM2 = np.load(SimMTM_folder+'avg_SimMTM/Gaussian_check_eeg_results_non_seizure_SimMTMfinallayer_N10k_100k.npy', allow_pickle=True).item()

N = [50,100,200,500]#,1000,5000,10000, 20000, 50000, 100000]

eeg = []
eegmulti = []
raw = []
stft = []
Hjorth = []
firstlayer = []
secondlayer = []
simMTM = []

cond = "non_seizure"

for ks_tuple in eeg1[cond]['ks']:
    eeg.append(ks_tuple[0])
    
for ks_tuple in eegmulti1[cond]['ks']:
    eegmulti.append(ks_tuple[0])

for ks_tuple in raw1[cond]['ks']:
    raw.append(ks_tuple[0])

for ks_tuple in stft1[cond]['ks']:
    stft.append(ks_tuple[0])

for ks_tuple in Hjorth1[cond]['ks']:
    Hjorth.append(ks_tuple[0])
    
for ks_tuple in firstlayer1[cond]['ks']:
    firstlayer.append(ks_tuple[0])
    
for ks_tuple in secondlayer1[cond]['ks']:
    secondlayer.append(ks_tuple[0])
    
for ks_tuple in simMTM1[cond]['ks']:
    simMTM.append(ks_tuple[0])

for ks_tuple in eeg2[cond]['ks']:
    eeg.append(ks_tuple[0])
    
for ks_tuple in eegmulti2[cond]['ks']:
    eegmulti.append(ks_tuple[0])

for ks_tuple in raw2[cond]['ks']:
    raw.append(ks_tuple[0])

for ks_tuple in stft2[cond]['ks']:
    stft.append(ks_tuple[0])
    
for ks_tuple in Hjorth2[cond]['ks']:
    Hjorth.append(ks_tuple[0])
    
for ks_tuple in firstlayer2[cond]['ks']:
    firstlayer.append(ks_tuple[0])
    
for ks_tuple in secondlayer2[cond]['ks']:
    secondlayer.append(ks_tuple[0])
    
for ks_tuple in simMTM2[cond]['ks']:
    simMTM.append(ks_tuple[0])

# lnm.append(lnm[-1])

print(len(firstlayer))
print(len(secondlayer))
print(len(Hjorth))
print(len(simMTM))



colors = {
    'CNN-1pat': 'royalblue',         # base CNN single-patient
    'CNN-multipat': 'navy',          # same family, darker blue
    'Raw': '#FFD700',        # Bright Yellow (Gold)
    'STFT': '#FFA500',       # Orange
    'Hjorth': '#8B4513',     # SaddleBrown (Dark Brownish Orange)
    'SimMTM-lastLayer': 'darkgreen', 
    'SimMTM-1stLayer': 'forestgreen',
    'SimMTM-2ndLayer': 'mediumseagreen',
    'Threshold': 'red'               # KS reference line
}

# Draw the eeg raw stft lnm with x as cond, in same figure
plt.figure(figsize=(10, 5))
plt.plot(N, eeg[0:len(N)], label='CNN-1 patient', marker='o', color=colors['CNN-1pat'])
plt.plot(N, eegmulti[0:len(N)], label='CNN-multipatients', marker='o', color=colors['CNN-multipat'])
# plt.plot(N, raw, label='Raw', marker='o', color=colors['Raw'])
# plt.plot(N, stft, label='STFT', marker='o', color=colors['STFT'])
# plt.plot(N, Hjorth, label='Hjorth', marker='o', color=colors['Hjorth'])
# plt.plot(N, firstlayer[0:len(N)], label='SimMTM-1stLayer', marker='o', color=colors['SimMTM-1stLayer'])

D_critical = 1.36 / np.sqrt(N)
plt.plot(N, D_critical, color=colors['Threshold'], linestyle='--', linewidth=3,
         label=r'$D_{\mathrm{critical}}$') # = \frac{1.36}{\sqrt{N}}


plt.plot(N, secondlayer[0:len(N)], label='SimMTM-2ndLayer', marker='o', color=colors['SimMTM-2ndLayer'])
plt.plot(N, simMTM[0:len(N)], label='SimMTM-lastLayer', marker='o', color=colors['SimMTM-lastLayer'])




# # Log scale for x-axis
# plt.xscale('log')
# plt.xticks(N, rotation=45)
plt.xlabel('N', fontsize=20)    
plt.ylabel(r'$D_N$', fontsize=20) 
plt.ylim(top=plt.ylim()[1] * 1.7)  # Increase y-axis upper limit by 15% 
plt.tick_params(axis='both', which='major', labelsize=20) 
plt.legend(loc='upper center', ncol=2, fontsize=21)
plt.grid()
plt.tight_layout()
plt.savefig('Gaussian_check_eeg_results.png')
