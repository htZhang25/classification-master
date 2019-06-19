import argparse
import numpy as np
import pandas as pd
from scipy.io import loadmat
import torch
from torch.autograd import Variable 
from torch.utils.data import DataLoader

from data import Dataset_Test
from model import Net1

from sklearn.metrics.ranking import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Testing!')
parser.add_argument('--model-path', metavar='DIR',
                    help='path to trained model', default='./best.pth.tar') 
parser.add_argument('--test-manifest', metavar='DIR',
                    help='path to validation manifest mat', default='./PRDataset/test_data.mat')                 
parser.add_argument('--save-path', metavar='DIR',
                    help='path to classification results txt', default='./result.txt')
args = parser.parse_args()

use_gpu = torch.cuda.is_available()

#load mat
test_mat =  loadmat(args.test_manifest)
test_data = pd.DataFrame(test_mat['yidali_test']).values

test_dataset = Dataset_Test(test_data)
test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False, num_workers=0) 

if use_gpu:
    model = Net1().cuda()
else:
    model = Net1()
model.load_state_dict(torch.load(args.model_path))
print('model:', model)

max_epoch = 1
for epoch in range(max_epoch):
    
    model.eval()
    running_corrects = 0
    total_num = 0
    class_out = np.zeros((9,9)).astype('int32')
    label_out = {}
    class_result = {}
    label_result = {}
    result = []
    f = open(args.save_path, 'w')
    for i, (data, label) in enumerate(test_loader, 0):
        if use_gpu:
            data = data.cuda()
        data, label = data.squeeze(0), label.squeeze(0)
        data, label = Variable(data), label.numpy()
        data = data.transpose(1, 0)
        output = model(data)

        predict = torch.max(output, 1)[1].data.cpu().numpy()      
        result.extend(predict)   

        #compute confusion matrix
        for c in range(predict.shape[0]):
            row = list(label[c, :]).index(1)
            class_out[row, predict[c]] += 1

            #total_num += 1

            if (list(label[c]).index(1)+1) not in label_out.keys():
                label_out[list(label[c]).index(1)+1] = 1
            else:
                label_out[list(label[c]).index(1)+1] += 1

        # copute accuracy
        for c in range(predict.shape[0]):
            if label[c, predict[c]] == 1:
                running_corrects += 1
                if (predict[c]+1) not in class_result.keys():
                    class_result[predict[c]+1] = 1
                else:
                    class_result[predict[c]+1] += 1
            total_num += 1
            if (list(label[c]).index(1)+1) not in label_result.keys():
                label_result[list(label[c]).index(1)+1] = 1				
            else:
                label_result[list(label[c]).index(1)+1] += 1
    print("\n")
    print("-------------------------------------------------------")
    print('testing accuracy: %.3f' % (float(running_corrects) / (total_num)))
    print('each num of class:')
    for c in label_result.keys():
        if c not in class_result.keys():
            class_result[c] = 0
        print(c, class_result[c], label_result[c])
        
    #compute kappa
    n = sum(map(sum,class_out))
    p0 = sum(class_out[i][i] for i in range(9))/n
    pe = (sum( sum(class_out[:,i])*sum(class_out[i,:])  for i in range(9)))/(n*n)
    k = (p0-pe)/(1-pe)
    print("\n")
    print("-------------------confusion matrix-------------------")
    print(class_out)
    print("kappa:", k)

    #save txt 
    for i in range(len(result)):
        f.write(str(result[i]))   
        f.write("\n")  
