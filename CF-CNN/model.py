import math
from tkinter.tix import Tree
import numpy as np
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.PReLU()
        
    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))

lookup_table = np.load('lookup_table.npy')
class CenterPooling(nn.Module):
    def __init__(self, input_size):
        super(CenterPooling, self).__init__()
        self.input_size = input_size
        self.n1, self.n2, self.n3 = self.get_n(self.input_size)

    def forward(self, x):
        out = []
        n3_up = x[:, :, :3 * math.ceil(self.n3 / 2), :]
        n3_row_max = nn.MaxPool2d((3, 1), stride=(3, 1))
        n3_up_out1 = n3_row_max(n3_up)
        out.append(n3_up_out1)
        n2_up = x[:, :, 3 * math.ceil(self.n3 / 2):3 * math.ceil(self.n3 / 2) + 2 * math.ceil(self.n2 / 2), :]
        n2_row_max = nn.MaxPool2d((2, 1), stride=(2, 1))
        n2_up_out1 = n2_row_max(n2_up)
        out.append(n2_up_out1)
        n1_up = x[:, :, 3 * math.ceil(self.n3 / 2) + 2 * math.ceil(self.n2 / 2):3 * math.ceil(self.n3 / 2) + 2 * math.ceil(self.n2 / 2) + math.ceil(self.n1 / 2), :]
        out.append(n1_up)
        index = 3 * math.ceil(self.n3 / 2) + 2 * math.ceil(self.n2 / 2) + math.ceil(self.n1 / 2)
        if ((self.n1 - math.ceil(self.n1 / 2)) > 0):
            n1_bottom = x[:, :, index:int(index + (self.n1 - math.ceil(self.n1 / 2))), :]
            index = int(index + (self.n1 - math.ceil(self.n1 / 2)))
            out.append(n1_bottom)
        if ((self.n2 - math.ceil(self.n2 / 2)) > 0):
            n2_bottom = x[:, : , index:int(index + 2 * (self.n2 - math.ceil(self.n2 / 2))), :]
            index = int(index + 2 * (self.n2 - math.ceil(self.n2 / 2)))
            n2_bottom_out = n2_row_max(n2_bottom)
            out.append(n2_bottom_out)
        if ((self.n3 - math.ceil(self.n3 / 2)) > 0):
            n3_bottom = x[:, :, index:int(index + 3 * (self.n3 - math.ceil(self.n3 / 2))), :]
            index = int(index + 3 * (self.n3 - math.ceil(self.n3 / 2)))
            n3_bottom_out = n3_row_max(n3_bottom)
            out.append(n3_bottom_out)
        concat = torch.cat(out, dim=2)
        out_1 = []
        n3_left = concat[:, :, :, :3 * math.ceil(self.n3 / 2)]
        n3_col_max = nn.MaxPool2d((1, 3), stride=(1, 3))
        n3_left_out = n3_col_max(n3_left)
        out_1.append(n3_left_out)
        n2_left = concat[:, :, :, 3 * math.ceil(self.n3 / 2):3 * math.ceil(self.n3 / 2) + 2 * math.ceil(self.n2 / 2)]
        n2_col_max = nn.MaxPool2d((1, 2), stride=(1, 2))
        n2_left_out = n2_col_max(n2_left)
        out_1.append(n2_left_out)
        n1_left = concat[:, :, :, 3 * math.ceil(self.n3 / 2) + 2 * math.ceil(self.n2 / 2):3 * math.ceil(self.n3 / 2) + 2 * math.ceil(self.n2 / 2) + math.ceil(self.n1 / 2)]
        out_1.append(n1_left)
        index = 3 * math.ceil(self.n3 / 2) + 2 * math.ceil(self.n2 / 2) + math.ceil(self.n1 / 2)
        if ((self.n1 - math.ceil(self.n1 / 2)) > 0):
            n1_right = concat[:, :, :, index:int(index + (self.n1 - math.ceil(self.n1 / 2)))]
            index = int(index + (self.n1 - math.ceil(self.n1 / 2)))
            out_1.append(n1_right)
        if ((self.n2 - math.ceil(self.n2 / 2)) > 0):
            n2_right = concat[:, :, :, index:int(index + 2 * (self.n2 - math.ceil(self.n2 / 2)))]
            index = int(index + 2 * (self.n2 - math.ceil(self.n2 / 2)))
            n2_right_out = n2_col_max(n2_right)
            out_1.append(n2_right_out)
        if ((self.n3 - math.ceil(self.n3 / 2)) > 0):
            n3_right = concat[:, :, :, index:int(index + 3 * (self.n3 - math.ceil(self.n3 / 2)))]
            index = int(index + 3 * (self.n3 - math.ceil(self.n3 / 2)))
            n3_right_out = n3_col_max(n3_right)
            out_1.append(n3_right_out)
        concat_1 = torch.cat(out_1, dim=3)
        return concat_1

    def get_n(self, input_size):
        n1 = input_size // 8
        n2 = input_size // 4
        n3 = input_size // 8
        residual = input_size - n1 - n2 * 2 - n3 * 3
        L = self.look_up(residual)
        n1 = n1 + L[1]
        n2 = n2 + L[2]
        n3 = n3 + L[3]
        assert(n1 + 2 * n2 + 3 * n3 == input_size)
        return n1, n2, n3

    def look_up(self, r):
        return lookup_table[:, r]
        
class Branch(nn.Module):
    def __init__(self, in_channels):
        super(Branch, self).__init__()
        self.conv1 = ConvBlock(in_channels, 36)
        self.conv2 = ConvBlock(36, 36)
        self.pool1 = CenterPooling(35)
        self.conv3 = ConvBlock(36, 48)
        self.conv4 = ConvBlock(48, 48)
        self.pool2 = CenterPooling(18)
        self.conv5 = ConvBlock(48, 68)
        self.conv6 = ConvBlock(68, 68)
        self.fc = nn.Linear(68 * 9 * 9, 300)
        self.norm = nn.BatchNorm1d(300)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.pool1(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.pool2(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = torch.reshape(out, (-1, 68 * 9 * 9))
        out = self.fc(out)
        out = self.norm(out)
        return out
        

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.branch_3d = Branch(3)
        self.branch_2d = Branch(2)
        self.fc = nn.Linear(600, 35 * 35)
        self.sig = nn.Sigmoid()

    def forward(self, img_3d, img_2d):
        out_3d = self.branch_3d(img_3d)
        out_2d = self.branch_2d(img_2d)
        out = self.fc(torch.cat((out_3d, out_2d), 1))
        out = torch.reshape(out, (-1, 1, 35, 35))
        out = self.sig(out)
        return out
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
        
    def load_model(self, path):
        self.load_state_dict(torch.load(path))