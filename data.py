import torch
from torch.utils import data
import numpy as np 

class Dataset_Test(data.Dataset):
    """docstring for Dataset"""
    def __init__(self, train_data):
        super(Dataset_Test, self).__init__()

        self.numclass = train_data.shape[1]

        for i in range(1, self.numclass+1):
            feature_i = train_data[0, i-1]
            if i == 1:
                self.features = feature_i.astype('float')
                self.labels = self.one_hot(i, feature_i.shape[1])
            else:
                self.features = np.concatenate((self.features,feature_i),axis=1)
                self.labels = np.concatenate((self.labels, self.one_hot(i, feature_i.shape[1])), axis=1)
        
        #normalization
        for feature_id in range(self.features.shape[0]):
            feature = self.features[feature_id]
            feature = (feature - feature.min()) / float(feature.max() - feature.min())
            self.features[feature_id, :] = feature
            
        self.total_num = self.features.shape[1]

    def __getitem__(self, index):
        data = self.features[:, index]
        label = self.labels[:, index]

        data = torch.from_numpy(data).float()
        label = torch.from_numpy(label).float()
        return data.unsqueeze(1), label

    def __len__(self):
        return self.features.shape[1]

    def one_hot(self, label, batch_size):
        binary = np.zeros((self.numclass, batch_size))
        binary[label-1, :] = 1

        return binary.astype("int")
