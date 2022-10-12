'''
Author: jyniki 1067087283@qq.com
Date: 2022-05-19 18:43:29
LastEditors: jyniki 1067087283@qq.com
LastEditTime: 2022-05-19 18:44:13
FilePath: /new_memae/lib/models/model_MVTec.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import cv2
from torchvision import transforms
import os
import torch

class MVTecDataset(Dataset):
    def __init__(self,root,split,w=256,h=256):
        """

        Args:
            root (path): ../data/mvtec_anomaly_detection/<object>/
            split (_type_): train or test
            w (int, optional): image w. Defaults to 256.
            h (int, optional): image h. Defaults to 256.
        """
        
        super(MVTecDataset,self).__init__()
        self.root = root
        self.split = split
        self.w = w
        self.h = h
        self.images = []
        self.labels = []
        self.setup()
        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([h,w]),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    def setup(self):
        if self.split == 'train':
            path = os.path.join(self.root,'train','good')
            train_images = os.listdir(path)
            for train_image in train_images:
                self.images.append(os.path.join(path,train_image))
                self.labels.append(0)
        elif self.split == 'test':
            path = os.path.join(self.root,'test')
            test_classes = os.listdir(path)
            for test_class in test_classes:
                first_path = os.path.join(path,test_class)
                test_images = os.listdir(first_path)
                for test_image in test_images:
                    test_image = os.path.join(first_path,test_image)
                    self.images.append(test_image)
                    if test_class == 'good':
                        self.labels.append(0)
                    else:
                        self.labels.append(1)
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,index):
        image = cv2.imread(self.images[index],cv2.IMREAD_COLOR)
        image = self.transformer(image)
        label = self.labels[index]
        
        label = torch.tensor(label)
        #print(type(label.item()))   #IndexError: invalid index of a 0-dim tensor. Use `tensor.item()` in Python or `tensor.item<T>()` in C++ to convert a 0-dim tensor to a number
        return image,label

