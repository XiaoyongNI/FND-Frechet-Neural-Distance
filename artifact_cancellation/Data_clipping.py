import scipy.io as scio
import numpy as np
from REST.specialFunction import *
from scipy.fft import fft
from scipy.signal import resample

def downsample_signal_vectorized(signal, fs, target_fs=128):
    """
    使用 resample 对多通道信号进行降采样。
    :param signal: 输入信号，形状为 [channels, timepoints]
    :param fs: 原始采样率
    :param target_fs: 目标采样率
    :return: 降采样后的信号
    """
    num_timepoints = signal.shape[1]
    target_points = int(num_timepoints * target_fs / fs)
    # 直接对整个数组降采样
    downsampled_signal = resample(signal, target_points, axis=1)
    return downsampled_signal

def data_process(pat_id, files_num, T=2, rate=100):
    infofile = '/net/inltitan1/scratch2/rpeng/long-term_dataset/ID'+pat_id+'/ID'+pat_id+'_info.mat'
    info = scio.loadmat(infofile)
    fs = info.get('fs')[0][0]
    seizure_begin = info.get('seizure_begin')
    seizure_end = info.get('seizure_end')
    print('fs: ', fs)
    print('seizure_begin: ', seizure_begin)
    print('seizure_end: ', seizure_end)
    window_size = T * fs
    overlap = int(0.5 * fs)

    seizure_list = []
    ictal_start, ictal_end = [],[]
    ictal_clips, ictal_feas = [],[]
    for t in range(len(seizure_begin)):
        srt = seizure_begin[t][0]
        end = seizure_end[t][0]
        print('srt: ', srt)
        print('end: ', end)
        ictal_id= int(srt / 3600) + 1
        offset = int((srt % 3600)*fs)
        ictal_id_end = int(end / 3600) + 1
        offset_end = int((end % 3600)*fs)
        print('ictal_id: ', ictal_id)
        print('ictal_id_end: ', ictal_id_end)

        if ictal_id == ictal_id_end:
            datafile = '/net/inltitan1/scratch2/rpeng/long-term_dataset/ID'+pat_id+'/ID'+pat_id+'_'+str(ictal_id)+'h.mat'
            data = scio.loadmat(datafile)
            print(data.keys())
            EEG = data.get('EEG')
            print('mat: ', np.shape(EEG))
            seizure_list.append(ictal_id)
            ictal_start.append(offset)
            ictal_end.append(offset_end)
        else:
            datafile1 = '/net/inltitan1/scratch2/rpeng/long-term_dataset/ID'+pat_id+'/ID'+pat_id+'_'+str(ictal_id)+'h.mat'
            datafile2 = '/net/inltitan1/scratch2/rpeng/long-term_dataset/ID'+pat_id+'/ID'+pat_id+'_'+str(ictal_id_end)+'h.mat'
            data1 = scio.loadmat(datafile1)
            data2 = scio.loadmat(datafile2)
            EEG = np.concatenate((data1.get('EEG'), data2.get('EEG')), axis = 1)
            offset_end = data1.get('EEG').shape[1] + offset_end
            seizure_list.append(ictal_id)
            seizure_list.append(ictal_id_end)
            ictal_start.append(offset)
            ictal_end.append(offset_end)

        slides_pos, fea_pos = [],[]
        for strt in range(offset, offset_end, window_size-overlap):
            if strt+window_size <= offset_end:
                clip = EEG[:,strt: strt+window_size]
                # clip = downsample_signal_vectorized(clip, fs, target_fs)
                clip_data = clip.reshape(clip.shape[0], T , int(clip.shape[1] / T))
                slides_pos.append(clip_data)
                fftSequence = np.abs(fft(clip_data,axis=2)[:,:,: int(fs/2)]) +1e-30
                fea_pos.append(fftSequence)

        ictal_clips.append(slides_pos)
        ictal_feas.append(fea_pos)

        
    ictal_dict_feas = {'seizure_id':seizure_list, 'ictal_clips':ictal_feas, 'offset_starts':ictal_start, 'offset_end':ictal_end}
    ictal_dict_data = {'seizure_id':seizure_list, 'ictal_clips':ictal_clips, 'offset_starts':ictal_start, 'offset_end':ictal_end}

    print(seizure_list)
    non_seizure_list = []
    non_ictal_clips, non_ictal_feas = [],[]
    for file in range(1,files_num):
        if file not in seizure_list:
            slides_neg, fea_neg =[], []
            non_seizure_list.append(file)
            datafile = '/net/inltitan1/scratch2/rpeng/long-term_dataset/ID'+pat_id+'/ID'+pat_id+'_'+str(file)+'h.mat'
            data = scio.loadmat(datafile)
            print(data.keys())
            EEG = data.get('EEG')
            print('mat: ', np.shape(EEG))
            length = EEG.shape[1]
            for strt in range(0, length, int(rate*window_size)):
                if strt+window_size <= length:
                    clip = EEG[:, strt: strt+window_size]
                    # clip = downsample_signal_vectorized(clip, fs, target_fs)
                    clip_data = clip.reshape(clip.shape[0], T , int(clip.shape[1]/T))
                    slides_neg.append(clip_data)
                    fftSequence = np.abs(fft(clip_data,axis=2)[:,:,: int(fs/2)]) +1e-30
                    fea_neg.append(fftSequence)
                    # print(fftSequence.shape)

            non_ictal_clips.append(slides_neg)
            non_ictal_feas.append(fea_neg)

    non_ictal_dict_data= {'non_seizure_id':non_seizure_list, 'non_ictal_clips':non_ictal_clips}
    non_ictal_dict_feas = {'non_seizure_id':non_seizure_list, 'non_ictal_clips':non_ictal_feas}

    print(non_seizure_list)

    feas_to_save = {'patient':pat_id, 'seizure_clips':ictal_dict_feas, 'non_seizure_clips':non_ictal_dict_feas}

    data_to_save = {'patient':pat_id, 'seizure_clips':ictal_dict_data, 'non_seizure_clips':non_ictal_dict_data}

    import pickle
    with open('/net/inltitan1/scratch2/rpeng/long-term_dataset/new_rate_'+str(rate)+'_T_'+str(T)+'s_Feas_of_pat_'+pat_id+'.pkl', 'wb') as f:
        pickle.dump(feas_to_save, f)

    with open('/net/inltitan1/scratch2/rpeng/long-term_dataset/new_rate_'+str(rate)+'_T_'+str(T)+'s_Clips_of_pat_'+pat_id+'.pkl', 'wb') as f:
        pickle.dump(data_to_save, f)

pid = ['01','02','03','04','05', '06', '07', '08','09','10','11','12','13','14','15','16','17','18']
lengths = [295,237,160,42,111,147,70,145,42,44,214,193,105,163,197,179,131,207]
print(len(pid))
print(len(lengths))
rates = [100, 50, 25, 10]
for rate in rates:
    for i in range(len(pid)):
        data_process(pid[i], lengths[i], T=4, rate=rate)

# data_process(pid[4], lengths[4])










# slides_neg,slides_pos = [],[]
# with frequecy features
# for i in range(1, 295):
#     print('hours: ', str(i))
#     datafile = '/scratch/rpeng/iEEG_ethz/long_term/long-term_dataset/ID'+pat_id+'/ID'+pat_id+'_'+str(i)+'h.mat'
#     data = scio.loadmat(datafile)
#     print(data.keys())
#     EEG = data.get('EEG')
#     print('mat: ', np.shape(EEG))


    
#     if i != ictal_id:
#         for strt in range(0, length, window_size - overlap):
#             if strt+window_size <= length:
#                 clip = EEG[:, strt: strt+window_size]
#                 clip_data = clip.reshape(clip.shape[0], 6 , int(clip.shape[1]/ 6))

#                 eeg_clip_new = np.zeros((clip_data.shape[0], clip_data.shape[1], CHANNEL_NUM))#channel number is the number of features
#                 fftSequence = np.abs(fft(clip_data,axis=2)[:,:,0: int(fs/2)]) +1e-30

#                 for i in range(0, clip_data.shape[0]):
#                     for j in range(0, clip_data.shape[1]):
#                         signal = clip_data[i, j, :]
#                         # envelope = getEnvelope(signal)
#                         fft_array = fftSequence[i, j, :]
#                         # LL = SF.getLineLength(signal)
#                         SSP = aggregation_List(fft_array, [0, 4, 8, 10, 30, 80, 127, 250, 500]) # https://doi.org/10.1016/j.nbd.2019.03.030 the ripple and fast ripple of HFOs
#                         eeg_clip_new[i, j] = SSP

#                 slides_neg.append(eeg_clip_new)


#     else:
#         for strt in range(offset, offset_end, window_size-overlap):
#             if strt+window_size <= offset_end:
#                 clip = EEG[:,strt: strt+window_size]
#                 clip_data = clip.reshape(clip.shape[0], 6 , int(clip.shape[1]/ 6))

#                 eeg_clip_new = np.zeros((clip_data.shape[0], clip_data.shape[1], CHANNEL_NUM)) #channel number is the number of features
#                 fftSequence = np.abs(fft(clip_data,axis=2)[:,:,0: int(fs/2)]) +1e-30
#                 for i in range(0, clip_data.shape[0]):
#                     for j in range(0, clip_data.shape[1]):
#                         signal = clip_data[i, j, :]
#                         # envelope = getEnvelope(signal)
#                         fft_array = fftSequence[i, j, :]
#                         # LL = SF.getLineLength(signal)
#                         SSP = aggregation_List(fft_array, [0, 4, 8, 10, 30, 80, 127, 250, 500]) # https://doi.org/10.1016/j.nbd.2019.03.030 the ripple and fast ripple of HFOs
#                         eeg_clip_new[i, j] = SSP

#                 slides_pos.append(eeg_clip_new)

# print(len(slides_neg))
# slides_neg = np.array(slides_neg)
# print(slides_neg.shape)

# print(len(slides_pos))
# slides_pos = np.array(slides_pos)
# print(slides_pos.shape)

# np.savez('/scratch/rpeng/iEEG_ethz/long_term/Clips_of_pat_'+pat_id+'.npz', patient = pat_id, seizure_clips_fea = slides_pos, non_seizure_clip_fea = slides_neg)


# with raw signals 
# for i in range(1, 295):
#     print('hours: ', str(i))
#     datafile = '/scratch/rpeng/iEEG_ethz/long_term/long-term_dataset/ID'+pat_id+'/ID'+pat_id+'_'+str(i)+'h.mat'
#     data = scio.loadmat(datafile)
#     print(data.keys())
#     EEG = data.get('EEG')
#     print('mat: ', np.shape(EEG))

#     window_size = 2 * fs
#     overlap = 1 * fs
#     length = EEG.shape[1]
    
#     if i != ictal_id:
#         for strt in range(0, length, window_size - overlap):
#             if strt+window_size <= length:
#                 clip = EEG[:, strt: strt+window_size]
#                 clip_data = clip.reshape(clip.shape[0], 2 , int(clip.shape[1]/ 2))
#                 fftSequence = np.abs(fft(clip_data,axis=2)[:,:,0: int(fs/2)]) +1e-30
#                 slides_neg.append(fftSequence)

#     else:
#         for strt in range(offset, offset_end, window_size-overlap):
#             if strt+window_size <= offset_end:
#                 clip = EEG[:,strt: strt+window_size]
#                 clip_data = clip.reshape(clip.shape[0], 2 , int(clip.shape[1]/ 2))
#                 fftSequence = np.abs(fft(clip_data,axis=2)[:,:,0: int(fs/2)]) +1e-30
#                 slides_pos.append(fftSequence)

# print(len(slides_neg))
# slides_neg = np.array(slides_neg)
# print(slides_neg.shape)

# print(len(slides_pos))
# slides_pos = np.array(slides_pos)
# print(slides_pos.shape)

# np.savez('/scratch/rpeng/iEEG_ethz/long_term/Clips_of_pat_'+pat_id+'.npz', patient = pat_id, seizure_clips_fea = slides_pos, non_seizure_clip_fea = slides_neg)
