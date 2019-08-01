import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import cv2
import os
import torch
import PIL.Image as Image
from config import get_config
conf = get_config(mode='training_eval')
def img_loader(path):
    try:
        with open(path, 'rb') as f:
            img = cv2.imread(path)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            
            return img
    except IOError:
        print('Cannot load image ' + path)

class VGG_FP(data.Dataset):
    def __init__(self, config, transform=None, loader=img_loader):
        self.root = config.train_root
        self.file_list = config.file_list
        self.transform = transform
        self.loader = loader
        image_list = []
        label_list = []
        with open(config.file_list) as f:
            img_label_list = f.read().splitlines()
        for info in img_label_list:
            image_path, label_name = info.split(' ')
            image_list.append(image_path)
            label_list.append(int(label_name))
        self.image_list = image_list
        self.label_list = label_list
        self.class_nums = len(np.unique(self.label_list))
        self.num_iter = len(self.image_list)// 64
        print("dataset size: ", len(self.image_list), '/', self.class_nums)
    def __getitem__(self, index):
        img_path = self.image_list[index]
        label = self.label_list[index]
        img = self.loader(os.path.join(self.root, img_path))
        # random flip with ratio of 0.5
        flip = np.random.choice(2) * 2 - 1
        img = img[:, ::flip, :]
        # img = (img - 127.5) / 128.0
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img)
        return img, label
        
    def __len__(self):
        return len(self.image_list)