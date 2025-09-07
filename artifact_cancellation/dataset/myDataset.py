import torch
from torch.utils.data import Dataset
import sklearn
import numpy as np

class iEEG_Dataset(Dataset):
    def __init__(self, data, labels, device):
        self.data = data
        self.labels = labels
        self.class_num = len(np.unique(self.labels)) #number of disease type be classified
        self.class_weight = sklearn.utils.class_weight.compute_class_weight('balanced', classes=np.array(range(self.class_num)), y= self.labels.numpy())#class weight
        self.sample_weight = sklearn.utils.class_weight.compute_sample_weight('balanced', self.labels.numpy())#sample weight
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx,:,:,:])
        label = torch.tensor(self.labels[idx])
       
        return data, label
