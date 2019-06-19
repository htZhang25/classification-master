import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_dim, input_dim).uniform_(-1., 1.), requires_grad = True)

    def forward(self, x):
        return x.mm(self.weight)

class Net1(torch.nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.attention = Attention(input_dim=103)
        self.conv1 = nn.Conv1d(in_channels=103, out_channels=206, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(in_channels=206, out_channels=412, kernel_size=1, padding=0)
        self.conv3 = nn.Conv1d(in_channels=412, out_channels=206, kernel_size=1, padding=0)
        self.conv4 = nn.Conv1d(in_channels=206, out_channels=103, kernel_size=1, padding=0)
        self.conv5 = nn.Conv1d(in_channels=103, out_channels=9, kernel_size=1, padding=0)

    def forward(self, x):   
        x = x.view(x.size(0), -1).permute(1, 0)
        x = self.attention(x.float())
        x = x.unsqueeze(2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
  
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = torch.sigmoid(self.conv5(x))
        return x.squeeze(2)

