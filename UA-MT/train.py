from cmath import log
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.nn.functional as F

import utils
from dataset import DataSet, RandomCrop, RandomRotFlip, ToTensor
from model import Net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epoch_num = 3
T = 8

def create_model(ema=False):
    net = Net(1, 2, 16)
    if ema:
        for param in net.parameters():
            param.detach_()
    return net

def get_current_consistency_weight(epoch):
    return 0.1 * utils.sigmod_rampup(epoch, epoch_num)

def update_emas(net, ema_net, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for param, ema_param in zip(net.parameters(), ema_net.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def train(net, ema_net, dataloader):
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    net.train()
    ema_net.train()
    for epoch in range(epoch_num):
        epoch_loss = 0
        iter_num = 0
        for i, sample in enumerate(dataloader):
            image, label = sample['image'], sample['label']
            image, label = image.to(device), label.to(device)
            unlabeled_image = image[2:]
            noise = torch.clamp(torch.randn_like(unlabeled_image) * 0.1, -0.2, 0.2)
            ema_input = unlabeled_image + noise
            output = net(image)
            with torch.no_grad():
                ema_output = ema_net(ema_input)
                
            image_ema = unlabeled_image.repeat(2, 1, 1, 1, 1)
            stride = image_ema.shape[0] // 2
            preds = torch.zeros([stride * T, 2, 112, 112, 80]).to(device)
            ema_input = torch.clamp(torch.randn_like(image_ema) * 0.1, -0.2, 0.2) + image_ema
            with torch.no_grad():
                for i in range(T // 2):
                    preds[2 * stride * i:2 * stride * (i + 1)] = ema_net(ema_input)
            preds = F.softmax(preds, dim=1)
            preds = preds.reshape([stride, T, 2, 112, 112, 80])
            preds = torch.mean(preds, dim=0)
            uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-10), dim=1, keepdim=True)
            
            loss_seg = F.cross_entropy(output[:2], label[:2])
            output_soft = F.softmax(output, dim=1)
            loss_seg_dice = utils.dice_loss(output_soft[:2, 1, :, :, :], label[:2] == 1)
            loss_supervised = 0.5 * (loss_seg + loss_seg_dice)

            consistency_weight = get_current_consistency_weight(epoch)
            consistency_loss = utils.softmax_mse_loss(output[2:], ema_output)
            H = (0.75 + 0.25 * utils.sigmod_rampup(epoch, epoch_num)) * np.log(2)
            mask = (uncertainty < H).float()
            consistency_loss = torch.sum(mask * consistency_loss) / (2.0 * torch.sum(mask) + 1e-10)
            
            loss = loss_supervised + consistency_weight * consistency_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_num += 1
            update_emas(net, ema_net, 0.99, iter_num)
            epoch_loss += loss.item()
        print('Epoch: {}, Loss: {}'.format(epoch, epoch_loss))
    net.save_model()
    
if __name__ == '__main__':
    net = create_model().to(device)
    ema_net = create_model(ema=True).to(device)
    dataset = DataSet(transform=transforms.Compose([RandomRotFlip(), RandomCrop((112, 112, 80)), ToTensor()]))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    train(net, ema_net, dataloader)