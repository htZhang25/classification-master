import torch
import pandas as pd
from scipy.io import loadmat
from data import Dataset, Dataset_Test
from torch.utils.data import DataLoader
from torch.autograd import Variable 
from model import Net1

use_gpu = torch.cuda.is_available()

#load mat
train_mat =  loadmat('./PRDataset/train_data.mat')
train_data = pd.DataFrame(train_mat['yidali_train']).values
test_mat =  loadmat('./PRDataset/test_data.mat')
test_data = pd.DataFrame(test_mat['yidali_test']).values


train_dataset = Dataset(train_data)
test_dataset = Dataset_Test(test_data)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False, num_workers=0)

model = Net1().cuda()
print(model)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001,betas=(0.9, 0.999))

epochs = 5000
best_acc = 0.
for epoch in range(epochs):
    print("epochs:", epoch)
    
    running_loss = 0.0
    for i, (data, label, weight) in enumerate(train_loader):
        if use_gpu:
            data, label, weight= data.cuda(), label.cuda(), weight.cuda()
        data, label, weight= data.squeeze(0), label.squeeze(0), weight.squeeze(0)
        data, label = Variable(data), Variable(label)
        optimizer.zero_grad()
        loss_func = torch.nn.BCELoss(weight = weight)
        out = model(data)

        loss = loss_func(out, label)
        loss.backward()
        optimizer.step()
        print("loss为：", loss)
        running_loss += loss.data
   
    model.eval()
    running_corrects = 0
    total_num = 0
    class_out = {}
    label_out = {}
    for i, (data, label) in enumerate(test_loader, 0):
        if use_gpu:
            data = data.cuda()
        #data, label = data.squeeze(0), label.squeeze(0)
        data, label = Variable(data), label.numpy()
        data = data.transpose(0,1)
        output = model(data)


        predict = torch.max(output, 1)[1].data.cpu().numpy()
        for c in range(predict.shape[0]):
            if label[c, predict[c]] == 1:
                running_corrects += 1
                if (predict[c]+1) not in class_out.keys():
                    class_out[predict[c]+1] = 1
                else:
                    class_out[predict[c]+1] += 1
            total_num += 1
            if (list(label[c]).index(1)+1) not in label_out.keys():
                label_out[list(label[c]).index(1)+1] = 1				
            else:
                label_out[list(label[c]).index(1)+1] += 1

    print('testing accuracy: %.3f' % (float(running_corrects) / (total_num)))
    print('each num of class:')
    for c in label_out.keys():
        if c not in class_out.keys():
            class_out[c] = 0
        print(c, class_out[c], label_out[c])
        
    acc= float(running_corrects) / (total_num)
    if best_acc < acc:
        best_acc = acc
        torch.save(model.state_dict(), './best.pth.tar')
    print('best_accuracy:', best_acc)
    

