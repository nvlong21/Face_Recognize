#!/usr/bin/env python
# encoding: utf-8
'''
@author: wujiyang
@contact: wujiyang@hust.edu.cn
@file: cfp.py
@time: 2018/12/26 16:19
@desc: the CFP-FP test dataset loader, it's similar with lfw and adedb, except that it has 700 pairs every fold
'''


import numpy as np
import cv2
import os
import torch.utils.data as data
import PIL.Image as Image
import torch
import torchvision.transforms as transforms

def img_loader(path):
    try:
        with open(path, 'rb') as f:
            img = cv2.imread(path)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            return img
    except IOError:
        print('Cannot load image ' + path)

class CFP_FP(data.Dataset):
    def __init__(self, config, transform=None, loader=img_loader):

        self.root = config.cfp_root
        self.file_list = config.cfp_file_list
        self.transform = transform
        self.loader = loader
        self.nameLs = []
        self.nameRs = []
        self.folds = []
        self.flags = []

        with open(config.cfp_file_list) as f:
            pairs = f.read().splitlines()
        for i, p in enumerate(pairs):
            p = p.split(' ')
            nameL = p[0]
            nameR = p[1]
            fold = i // 700
            flag = int(p[2])

            self.nameLs.append(nameL)
            self.nameRs.append(nameR)
            self.folds.append(fold)
            self.flags.append(flag)

    def __getitem__(self, index):

        img_l = self.loader(os.path.join(self.root, self.nameLs[index]))
        img_r = self.loader(os.path.join(self.root, self.nameRs[index]))
        imglist = [img_l, cv2.flip(img_l, 1), img_r, cv2.flip(img_r, 1)]

        if self.transform is not None:
            for i in range(len(imglist)):
                imglist[i] = Image.fromarray(imglist[i].astype('uint8'), 'RGB')
                imglist[i] = self.transform(imglist[i])

            imgs = imglist
            return imgs
        else:
            imgs = [torch.from_numpy(i) for i in imglist]
            return imgs

    def __len__(self):
        return len(self.nameLs)

