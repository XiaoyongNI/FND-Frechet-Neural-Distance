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

N = [50,100,200,500,1000,5000,10000, 20000, 50000, 100000]

eeg = []
eegmulti = []
raw = []
stft = []
Hjorth = []
firstlayer = []
secondlayer = []
simMTM = []
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
    
for ks_tuple in firstlayer1[cond]['var']:
    firstlayer.append(ks_tuple[0])
    
for ks_tuple in secondlayer1[cond]['var']:
    secondlayer.append(ks_tuple[0])
    
for ks_tuple in simMTM1[cond]['var']:
    simMTM.append(ks_tuple[0])

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
    
for ks_tuple in firstlayer2[cond]['var']:
    firstlayer.append(ks_tuple[0])
    
for ks_tuple in secondlayer2[cond]['var']:
    secondlayer.append(ks_tuple[0])
    
for ks_tuple in simMTM2[cond]['var']:
    simMTM.append(ks_tuple[0])



colors = {
    'CNN-1pat': 'royalblue',         # base CNN single-patient
    'CNN-multipat': 'navy',          # same family, darker blue
    'Raw': '#FFD700',        # Bright Yellow (Gold)
    'STFT': '#FFA500',       # Orange
    'Hjorth': '#8B4513',     # SaddleBrown (Dark Brownish Orange)
    'SimMTM-lastLayer': 'darkgreen', 
    'SimMTM-1stLayer': 'lightgreen',
    'SimMTM-2ndLayer': 'mediumseagreen',
    'Threshold': 'red'               # KS reference line
}


# Draw the eeg raw stft lnm with x as cond, in same figure
plt.figure(figsize=(10, 5))
plt.plot(N, eeg, label=r'CNN-1 patient$||\Sigma||$', marker='o', color=colors['CNN-1pat'])
plt.plot(N, eegmulti, label=r'CNN-multipatients$||\Sigma||$', marker='o', color=colors['CNN-multipat'])
# plt.plot(N, raw, label='Raw', marker='o', color=colors['Raw'])
# plt.plot(N, stft, label='STFT', marker='o', color=colors['STFT'])
# plt.plot(N, Hjorth, label='Hjorth', marker='o', color=colors['Hjorth'])
# plt.plot(N, firstlayer, label='SimMTM-1stLayer', marker='o', color=colors['SimMTM-1stLayer'])
plt.plot(N, secondlayer, label=r'SimMTM-2ndLayer$||\Sigma||$', marker='o', color=colors['SimMTM-2ndLayer'])
plt.plot(N, simMTM, label=r'SimMTM-lastLayer$||\Sigma||$', marker='o', color=colors['SimMTM-lastLayer'])

#Log
plt.yscale('log')
# plt.xticks(N, rotation=45)
plt.xlabel('N', fontsize=20)    
plt.ylabel(r'$||\Sigma||$', rotation=90, fontsize=21) 
plt.ylim(top=plt.ylim()[1] * 10)  # Increase y-axis upper limit 
plt.tick_params(axis='both', which='major', labelsize=21) 
plt.legend(loc='upper center', ncol=2, fontsize=19)
plt.grid()
plt.tight_layout()
plt.savefig('Var_check_eeg_results.png')
