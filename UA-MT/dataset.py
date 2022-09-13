import os
import torch
import numpy as np
from torch.utils.data import Dataset
import h5py

class DataSet(Dataset):
    def __init__(self, transform, split='train'):
        self.image_list = []
        self.transform = transform
        file_list_path = os.path.join('dataset', split + '.list')
        with open(file_list_path, 'r') as f:
            for line in f:
                self.image_list.append(os.path.join('dataset', os.path.join(line.strip(), 'mri_norm2.h5')))
                
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        content = h5py.File(self.image_list[index], 'r')
        image = content['image'][:]
        label = content['label'][:]
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample
    
class RandomCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size
        
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        w, h, d = image.shape
        if w <= self.output_size[0] or h <= self.output_size[1] or d <= self.output_size[2]:
            pw = (self.output_size[0] - w) // 2 + 3
            ph = (self.output_size[1] - h) // 2 + 3
            pd = (self.output_size[2] - d) // 2 + 3
            image = np.pad(image, ((pw, pw), (ph, ph), (pd, pd)), 'constant', constant_values=0)
            label = np.pad(label, ((pw, pw), (ph, ph), (pd, pd)), 'constant', constant_values=0)
        w, h, d = image.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return {'image': image, 'label': label}

class RandomRotFlip(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis).copy()
        label = np.flip(label, axis).copy()
        return {'image': image, 'label': label}

class ToTensor(object):
    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        image = torch.from_numpy(image)
        label = torch.from_numpy(sample['label']).long()
        return {'image': image, 'label': label}
