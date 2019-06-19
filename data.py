import torch
from torch.utils import data
import numpy as np 


class Dataset(data.Dataset):
    def __init__(self, train_data):
        super(Dataset, self).__init__()
        self.train_data = train_data.squeeze(0)
        self.numclass = train_data.shape[1]
        self.number_each = []

        num_batch = 100
        mini_each = []
        for i in range(self.numclass):
            self.train_data[i] = self.train_data[i].astype('float')
            data = self.train_data[i].astype('float')
            # self.train_data[i] = data     
            self.number_each.append(data.shape[1])
            
            mini_each.append(data.shape[1]//num_batch)
        # self.train_data = self.train_data.astype('float')
        #normalization
        for id in range(self.train_data[0].shape[0]):
            id_max = 0
            id_min = 10000
            for ii in range(self.numclass):
                data = self.train_data[ii][id, :].astype('float')
                if data.max() > id_max:
                    id_max = data.max()
                if data.min() < id_min:
                    id_min = data.min()
            for ii in range(self.numclass):

                data = self.train_data[ii][id, :].astype('float')   
                
                data = (data - id_min) / float(id_max - id_min)
                self.train_data[ii][id, :] = data
        self.inputs = []
        self.labels = []
        self.one_hot_labels = []
        self.weight = []

        #shuffle
        for i in range(self.numclass):
            feature_i = self.train_data[i]
            arr = np.arange(feature_i.shape[1])
            np.random.shuffle(arr)
            self.train_data[i] = feature_i[:,arr]

        for j in range(0, num_batch):  
            b = np.zeros((103, 0)).astype('float')
            l = []  
            w = []
            for i in range(self.numclass):
                # weight = 100/mini_each[i]
                weight = sum(mini_each)/mini_each[i]
                w.extend([weight]*mini_each[i]) 
                a = self.train_data[i][:,j*mini_each[i]:(j+1)*mini_each[i]]
                b = np.concatenate((b,a),axis=1)  
                l.extend([i]*mini_each[i])
            self.inputs.append(b) 
            self.labels.append(l)
            self.weight.append(w)
        self.inputs = np.array(self.inputs)
        self.labels = np.array(self.labels)
        self.weight = np.array(self.weight)

        for i in range(len(self.labels)):  
            label = torch.LongTensor(self.labels[i])
            label = label.reshape(-1, 1)
            batch_size = label.shape[0]
            one_hot_labels = torch.zeros(batch_size, self.numclass).scatter_(1, label, 1)
            self.one_hot_labels.append(one_hot_labels)

    def __getitem__(self, index):
        data = torch.from_numpy(self.inputs[index])#.permute(1, 0).unsqueeze(2)
        label = self.one_hot_labels[index]
        weight = torch.from_numpy(self.weight[index]).float().unsqueeze(1)

        #shuffle
        arr = np.arange(data.shape[1])
        np.random.shuffle(arr)
        data = data[:, arr]
        label = label[arr, :]
        weight = weight[arr, :]
        
        return data, label, weight

    def __len__(self):
        return len(self.inputs)

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
