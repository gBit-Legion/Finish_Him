import torch
import torch.nn as nn
import torch.nn.functional as F

class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super(MyNeuralNetwork, self).__init__()

        self.conv1 = nn.Conv2d(3, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.drop1 = nn.Dropout(p=0.15)
        self.bn3 = nn.BatchNorm2d(128)
        self.lstm1 = nn.LSTM(128 * 52 * 70, 128, batch_first=True)
        self.bn4 = nn.BatchNorm1d(128)
        self.lstm2 = nn.LSTM(128, 128, batch_first=True)
        self.dense1 = nn.Linear(128 * 24, 32)
        self.dense2 = nn.Linear(32, 32)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, wrf_data):
        wrf_data = F.relu(self.bn1(self.conv1(wrf_data)))
        wrf_data = F.relu(self.bn2(self.conv2(wrf_data)))
        wrf_data = F.relu(self.conv3(wrf_data))
        wrf_data = self.drop1(wrf_data)
        wrf_data = self.bn3(wrf_data)
        wrf_data, _ = self.lstm1(wrf_data)
        wrf_data = self.bn4(wrf_data)
        wrf_data, _ = self.lstm2(wrf_data)
        wrf_data = wrf_data.contiguous().view(wrf_data.size(0), -1)
        wrf_data = F.relu(self.dense1(wrf_data))
        wrf_data = F.relu(self.dense2(wrf_data))
        wrf_data = self.softmax(wrf_data)

        return wrf_data