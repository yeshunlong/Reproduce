import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from dataset import DataSet
from model import Net
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_iou(pred, target, thread_hold):
    inserection = (pred > thread_hold) & (target > thread_hold)
    union = (pred > thread_hold) | (target > thread_hold)
    iou = inserection.sum() / union.sum()
    return iou

def test(net, dataloader):
    net.eval()
    with torch.no_grad():
        for img_3d, img_2d, mask in dataloader:
            img_3d = img_3d.to(device)
            img_2d = img_2d.to(device)
            mask = mask.to(device)
            output = net(img_3d, img_2d)
            output = torch.squeeze(output).detach().cpu().numpy()
            mask = torch.squeeze(mask).cpu().numpy()
            plt.subplot(1, 2, 1)
            plt.title('Pridiction IOU = {}'.format(get_iou(output, mask, 0.5)))
            plt.imshow(output, 'gray')
            plt.subplot(1, 2, 2)
            plt.imshow(mask, 'gray')
            plt.show()

if __name__ == '__main__':
    test_set = DataSet('./dataset/test/')
    test_dataloader = torch.utils.data.DataLoader(test_set)
    net = Net().to(device)
    net.load_model('./model.pth')
    test(net, test_dataloader)
    