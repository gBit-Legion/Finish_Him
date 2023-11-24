from torch import nn
import torch
import torch.nn.functional as F


class Correntor(nn.Module):
    def __init__(self):
        super(Correntor, self).__init__()

        self.conv1 = nn.Conv2d(3, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        # self.conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(128)
        # self.conv3 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1)
        # self.drop1 = nn.Dropout(p=0.15)
        # self.bn3 = nn.BatchNorm2d(128)
        self.lstm1 = nn.LSTM(58800, 128, batch_first=False)
        self.bn4 = nn.BatchNorm1d(128)
        # self.lstm2 = nn.LSTM(128, 128, batch_first=True)
        self.dense1 = nn.Linear(16384, 8192)
        self.dense2 = nn.Linear(8192, 3*210*280)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, wrf_data):
        wrf_data = F.relu(self.bn1(self.conv1(wrf_data)))
        # wrf_data = F.relu(self.bn2(self.conv2(wrf_data)))
        # wrf_data = F.relu(self.conv3(wrf_data))
        # wrf_data = self.drop1(wrf_data)
        # wrf_data = self.bn3(wrf_data)
        wrf_data = wrf_data.view(wrf_data.size(0), wrf_data.size(1), -1)
        # wrf_data = wrf_data.reshape(wrf_data.size(0), wrf_data.size(1), wrf_data.size(2) + wrf_data.size(3))
        wrf_data, _ = self.lstm1(wrf_data)
        wrf_data = self.bn4(wrf_data)
        # wrf_data, _ = self.lstm2(wrf_data)
        wrf_data = wrf_data.contiguous().view(wrf_data.size(0), -1)
        wrf_data = F.relu(self.dense1(wrf_data))
        wrf_data = F.relu(self.dense2(wrf_data))
        wrf_data = self.softmax(wrf_data)
        # wrf_data = wrf_data.view(batch_size, -1, 2)
        return wrf_data


class ConstantBias(nn.Module):
    def __init__(self, channels=6):
        super().__init__()
        self.channels = channels
        self.linear = nn.Linear(channels*210*280, 3)

    def forward(self, x):
        '''
        input: S*B*C*H*W
        :param input:
        :return:
        '''
        output = self.linear(x.view(*x.shape[:-3], self.channels*210*280))
        o_input = torch.split(x, 3, dim=-3)
        output = o_input[0].permute(3, 4, 0, 1, 2) + output
        return output.permute(2, 3, 4, 0, 1)
