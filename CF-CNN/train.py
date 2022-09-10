import copy
import torch
import torch.nn as nn

from dataset import DataSet
from model import Net
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test(net, dataloader):
    criterion = nn.BCELoss()
    test_lost = 0
    net.eval()
    with torch.no_grad():
        for img_3d, img_2d, mask in dataloader:
            img_3d = img_3d.to(device)
            img_2d = img_2d.to(device)
            mask = mask.to(device)
            output = net(img_3d, img_2d)
            loss = criterion(output, mask)
            test_lost += loss.item()
    print('test loss: {}'.format(test_lost))

def train(net, train_dataloader, test_dataloader, epoches=800):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    for epoch in range(epoches):
        epoch_loss = 0
        for img_3d, img_2d, mask in train_dataloader:
            optimizer.zero_grad()
            img_3d = img_3d.to(device)
            img_2d = img_2d.to(device)
            mask = mask.to(device)
            output = net(img_3d, img_2d)
            loss = criterion(output, mask)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print('epoch: {}/{}, train loss: {}'.format(epoch + 1, epoches, epoch_loss))
        test(copy.deepcopy(net), test_dataloader)
    net.save_model('./model.pth')

if __name__ == '__main__':
    train_set = DataSet('./dataset/train/')
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4)
    test_set = DataSet('./dataset/test/')
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=8, shuffle=False, num_workers=4)
    net = Net().to(device)
    train(net, train_dataloader, test_dataloader)
    