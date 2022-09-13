import imp


import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, stages, in_channels, out_channels, kernel_size, stride, padding, upsample=False):
        super().__init__()
        ops = []
        for i in range(stages):
            if i == 0:
                in_channels = in_channels
            else:
                in_channels = out_channels
            if upsample == False:
                ops.append(nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding))
            else:
                ops.append(nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding))
            ops.append(nn.BatchNorm3d(out_channels))
            ops.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*ops)
        
    def forward(self, x):
        return self.layers(x)

class Net(nn.Module):
    def __init__(self, in_channels, out_channels, n_filters):
        super().__init__()
        self.conv = nn.Conv3d(n_filters, out_channels, 1, 1, 0)
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        self.conv1 = ConvBlock(1, in_channels, n_filters, 3, 1, 1)
        self.conv2 = ConvBlock(1, n_filters, n_filters * 2, 2, 2, 0)
        self.conv3 = ConvBlock(2, n_filters * 2, n_filters * 2, 3, 1, 1)
        self.conv4 = ConvBlock(1, n_filters * 2, n_filters * 4, 2, 2, 0)
        self.conv5 = ConvBlock(3, n_filters * 4, n_filters * 4, 3, 1, 1)
        self.conv6 = ConvBlock(1, n_filters * 4, n_filters * 8, 2, 2, 0)
        self.conv7 = ConvBlock(3, n_filters * 8, n_filters * 8, 3, 1, 1)
        self.conv8 = ConvBlock(1, n_filters * 8, n_filters * 16, 2, 2, 0)
        self.conv9 = ConvBlock(3, n_filters * 16, n_filters * 16, 3, 1, 1)
        self.dconv1 = ConvBlock(1, n_filters * 16, n_filters * 8, 2, 2, 0, upsample=True)
        self.dconv2 = ConvBlock(3, n_filters * 8, n_filters * 8, 3, 1, 1)
        self.dconv3 = ConvBlock(1, n_filters * 8, n_filters * 4, 2, 2, 0, upsample=True)
        self.dconv4 = ConvBlock(3, n_filters * 4, n_filters * 4, 3, 1, 1)
        self.dconv5 = ConvBlock(1, n_filters * 4, n_filters * 2, 2, 2, 0, upsample=True)
        self.dconv6 = ConvBlock(2, n_filters * 2, n_filters * 2, 3, 1, 1)
        self.dconv7 = ConvBlock(1, n_filters * 2, n_filters * 1, 2, 2, 0, upsample=True)
        self.dconv8 = ConvBlock(1, n_filters, n_filters, 3, 1, 1)
        
    def encoder(self, x):
        x1 = self.conv1(x)
        x2 = self.conv3(self.conv2(x1))
        x3 = self.conv5(self.conv4(x2))
        x4 = self.conv7(self.conv6(x3))
        x5 = self.conv9(self.conv8(x4))
        x5 = self.dropout(x5)
        return [x1, x2, x3, x4, x5]
    
    def decoder(self, x):
        [x1, x2, x3, x4, x5] = x
        x5 = self.dconv1(x5)
        x6 = self.dconv3(self.dconv2(x5 + x4))
        x7 = self.dconv5(self.dconv4(x6 + x3))
        x8 = self.dconv7(self.dconv6(x7 + x2))
        x9 = self.dconv8(x8 + x1)
        x9 = self.dropout(x9)
        out = self.conv(x9)
        return out
        
    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out
    
    def save_model(self):
        torch.save(self.state_dict(), 'model.pth')
        
    def load_model(self):
        self.load_state_dict(torch.load('model.pth'))