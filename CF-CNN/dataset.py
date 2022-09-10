import imp
import os
from PIL import Image
import torch
from torchvision.transforms import transforms

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

class DataSet:
    def __init__(self, path):
        self.imgs = []
        for file in os.listdir(path):
            file_path = path + file
            img = Image.open(file_path)
            if 'mask' in file:
                img = Image.open(file_path).convert('1')
                img = transforms.ToTensor()(img)
            else:
                img = data_transform(img)
            self.imgs.append(img)

    def __getitem__(self, index):
        img_1_35, img_2_35, img_2_65, img_3_35, mask, _ = self.imgs[6 * index], self.imgs[6 * index + 1], self.imgs[6 * index + 2], self.imgs[6 * index + 3], self.imgs[6 * index + 4], self.imgs[6 * index + 5]
        img_3d = torch.cat((img_1_35, img_2_35, img_3_35), 0)
        img_2d = torch.cat((img_2_35, img_2_65), 0)
        return img_3d, img_2d, mask
    
    def __len__(self):
        return len(self.imgs) // 6
