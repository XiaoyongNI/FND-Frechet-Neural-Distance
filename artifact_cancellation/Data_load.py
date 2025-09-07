import pickle
import random
import os
import numpy as np

def load_data(rate, T, patient_ids, data_dir):
    """
    Load seizure and non-seizure clips for the given patients.
    """
    data = {}

    for pat_id in patient_ids:
        clips_path = os.path.join(data_dir, f'stride01_new_rate_{rate}_T_{T}s_Clips_of_pat_{pat_id}.pkl')
        print(f"Loading data from {clips_path}...")
        
        if not os.path.exists(clips_path):
            print(f"Warning: Data for patient {pat_id} not found. Skipping.")
            continue
        
        with open(clips_path, 'rb') as f:
            clips_data = pickle.load(f)
        print(clips_data['data'][0].shape)

        labels = np.asarray(clips_data['labels'])  # 保证是array，不是标量

        
        idx_0 = np.where(labels == 0)[0]
        idx_1 = np.where(labels == 1)[0]

        selected_data1 = [np.expand_dims(clips_data['data'][i], axis = 0) for i in idx_1]
        selected_data0 = [np.expand_dims(clips_data['data'][i], axis = 0) for i in idx_0]

        seizure_data = np.concatenate(selected_data1, axis=0).squeeze()
        non_seizure_data = np.concatenate(selected_data0, axis=0).squeeze()

        # seizure_data = seizure_data.reshape(seizure_data.shape[0], 88, -1)
        # non_seizure_data = non_seizure_data.reshape(non_seizure_data.shape[0], 88, -1)

        print(f"Seizure data shape: {seizure_data.shape}")
        print(f"Non-seizure data shape: {non_seizure_data.shape}")
        
        data[pat_id] = {
            'seizure_clips': seizure_data,
            'non_seizure_clips': non_seizure_data
        }
    
    return data

def uniform_sample(clips, num_samples):
    clips = np.array(clips)
    n = len(clips)
    if num_samples >= n:
        return clips.tolist()
    
    # 计算采样的索引位置
    indices = np.linspace(0, n-1, num_samples, dtype=int)
    return clips[indices]

def select_equal_clips(data, num_clips):
    """
    Randomly select equal number of seizure and non-seizure clips.
    """
    selected_data = {}
    
    for pat_id, clips in data.items():
        seizure_clips_data = clips['seizure_clips']
        # seizure_clips_data = seizure_clips['ictal_clips'][0]
        non_seizure_clips_data = clips['non_seizure_clips']
        # non_seizure_clips_data = non_seizure_clips['non_ictal_clips'][0]
                  
        
        if seizure_clips_data.shape[0] < num_clips or non_seizure_clips_data.shape[0] < num_clips:
            print(f"Warning: Not enough clips for patient {pat_id}. Using minimum available.")
            min_clips = min(len(seizure_clips_data.shape[0]), len(non_seizure_clips_data.shape[0]))
        else:
            min_clips = num_clips
        
        selected_data[pat_id] = {
            'seizure_clips': uniform_sample(seizure_clips_data, min_clips),
            'non_seizure_clips': np.array(random.sample(non_seizure_clips_data.tolist(), min_clips))
        }
    
    return selected_data

if __name__ == "__main__":
    rate = 0  # Change as needed
    T = 1  # Change as needed
    patient_ids = [ '01','02','03','05','07','15']  # Patients to include
    data_dir = '/net/inltitan1/scratch2/rpeng/long-term_dataset/'
    num_clips = 100  # Number of clips per class per patient
    save_dir = '/net/inltitan1/scratch2/yuhxie/ethz_data'
    # Load data
    data = load_data(rate, T, patient_ids, data_dir)
    
    # Select equal clips
    selected_data = select_equal_clips(data, num_clips)
    print(selected_data['01']['seizure_clips'].shape)
    # Save selected data
    with open(os.path.join(save_dir, 'patall512_100clips_1s.pkl'), 'wb') as f:
        pickle.dump(selected_data, f)
    
    print("Selected clips saved successfully!")
