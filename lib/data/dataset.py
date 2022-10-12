import numpy as np
import torch
from torchvision import datasets
import cv2
import os
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets
from torchvision import transforms

class Data():
    def __init__(self,train_dataloader,test_dataloader):
        self.train = train_dataloader
        self.test = test_dataloader

def get_dataset(cfg):
    if cfg.data_type == 'cifar10':
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        train_dataset = datasets.CIFAR10(root='../data',train=True,transform=transformer)
        test_dataset = datasets.CIFAR10(root='../data',train=False,transform=transformer)
    elif cfg.data_type == 'mnist':
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(root='../data',train=True,transform=transformer)
        test_dataset = dataset.MNIST(root='../data',train=False,transform=transformer)
    elif cfg.data_type == 'MVTec':
        train_dataset = MVTecDataset('../data',split='train')
        test_dataset = MVTecDataset('../data',split='test')
    elif cfg.data_type == 'CIG':
        train_dataset = CIG_Dataset('../data/cropped_image',split='train',w=416,h=416)
        test_dataset = CIG_Dataset('../data/cropped_image',split='test',w=416,h=416)
    return train_dataset,test_dataset
        
def get_cifar_anomaly_dataset(train_dataset,test_dataset,nrm_cls_idx):
    
    train_imgs,train_labels = train_dataset.data,np.array(train_dataset.targets)
    test_imgs,test_labels = test_dataset.data,np.array(test_dataset.targets)
  
    nrm_trn_idx = np.where(train_labels == nrm_cls_idx)[0]
    abn_trn_idx = np.where(train_labels != nrm_cls_idx)[0]
    nrm_trn_img = train_imgs[nrm_trn_idx]
    abn_trn_img = train_imgs[abn_trn_idx]
    nrm_trn_lbl = train_labels[nrm_trn_idx]
    abn_trn_lbl = train_labels[abn_trn_idx]



    nrm_tst_idx = np.where(test_labels == nrm_cls_idx)[0]
    abn_tst_idx = np.where(test_labels != nrm_cls_idx)[0]
    nrm_tst_img = test_imgs[nrm_tst_idx]
    abn_tst_img = test_imgs[abn_tst_idx]
    nrm_tst_lbl = test_labels[nrm_tst_idx]
    abn_tst_lbl = test_labels[abn_tst_idx]

    #对正常类分配标签0 对异常类分配标签1
    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1
    abn_tst_lbl[:] = 1

    
    #对dataset进行改变
    train_dataset.data = np.copy(nrm_trn_img)
    train_dataset.targets = np.copy(nrm_trn_lbl)
    test_dataset.data = np.concatenate((abn_trn_img, nrm_tst_img, abn_tst_img),axis=0)
    test_dataset.targets = np.concatenate((abn_trn_lbl,nrm_tst_lbl, abn_tst_lbl),axis=0)
    
    return train_dataset,test_dataset

def get_mnist_anomaly_dataset(train_dataset,test_dataset,nrm_cls_idx):
    
    train_imgs,train_labels = train_dataset.data,train_dataset.targets
    test_imgs,test_labels = test_dataset.data,test_dataset.targets


    nrm_trn_idx = torch.from_numpy(np.where(train_labels == nrm_cls_idx)[0])
    abn_trn_idx = torch.from_numpy(np.where(train_labels != nrm_cls_idx)[0])


    nrm_trn_img = train_imgs[nrm_trn_idx]
    nrm_trn_lbl = train_labels[nrm_trn_idx]
    abn_trn_img = train_imgs[abn_trn_idx]
    abn_trn_lbl = train_labels[abn_trn_idx]
    
    nrm_tst_idx = torch.from_numpy(np.where(test_labels == nrm_cls_idx)[0]) # np.where() 这里是tensor类型 内部的数据格式是bool类型

    abn_tst_idx = torch.from_numpy(np.where(test_labels != nrm_cls_idx)[0])
    nrm_tst_img = test_imgs[nrm_tst_idx]
    nrm_tst_lbl = test_labels[nrm_tst_idx]
    abn_tst_img = test_imgs[abn_tst_idx]
    abn_tst_lbl = test_labels[abn_tst_idx]

    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1 
    abn_tst_lbl[:] = 1 
    
    train_dataset.data = nrm_trn_img.clone()
    train_dataset.targets = nrm_trn_lbl.clone()
    test_dataset.data = torch.cat((abn_trn_img, nrm_tst_img, abn_tst_img),axis=0)
    test_dataset.targets = torch.cat((abn_trn_lbl,nrm_tst_lbl,abn_tst_lbl),axis=0)   

    return train_dataset,test_dataset

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
    
def get_cifar_data(train_dataset,test_dataset,user_id):
    '''
    在cifar10数据集下,0-4为一个边端见到的数据,5-9为另一个边端见到的数据 
    设定边端一下的正常样本是4,边端二下的正常样本是6
    '''
    train_imgs,train_labels = train_dataset.data,np.array(train_dataset.targets)
    test_imgs,test_labels = test_dataset.data,np.array(test_dataset.targets)    
    
    if user_id == 1:
        
        nrm_trn_idx = np.where(train_labels==4)[0]
        abn_trn_idx = np.concatenate([np.where(train_labels==1)[0],np.where(train_labels==2)[0],np.where(train_labels==3)[0],np.where(train_labels==0)[0]])
        nrm_tst_idx = np.where(test_labels==4)[0]
        abn_tst_idx = np.concatenate([np.where(test_labels==1)[0],np.where(test_labels==2)[0],np.where(test_labels==3)[0],np.where(test_labels==0)[0]])
        
    elif user_id == 2:

        nrm_trn_idx = np.where(train_labels==6)[0]
        abn_trn_idx = np.concatenate([np.where(train_labels==5)[0],np.where(train_labels==7)[0],np.where(train_labels==8)[0],np.where(train_labels==9)[0]])
        nrm_tst_idx = np.where(test_labels==6)[0]
        abn_tst_idx = np.concatenate([np.where(test_labels==5)[0],np.where(test_labels==7)[0],np.where(test_labels==8)[0],np.where(test_labels==9)[0]])     

    nrm_trn_imgs = train_imgs[nrm_trn_idx]
    nrm_trn_lbls = train_labels[nrm_trn_idx]
    abn_trn_imgs = train_imgs[abn_trn_idx]
    abn_trn_lbls = train_labels[abn_trn_idx]
    nrm_tst_imgs = test_imgs[nrm_tst_idx]
    nrm_tst_lbls = test_labels[nrm_tst_idx]
    abn_tst_imgs = test_imgs[abn_tst_idx]
    abn_tst_lbls = test_labels[abn_tst_idx]

    nrm_trn_lbls[:] = 0
    abn_trn_lbls[:] = 1
    nrm_tst_lbls[:] = 0
    abn_tst_lbls[:] = 1
    
    train_dataset.data = np.copy(nrm_trn_imgs)
    train_dataset.targets = np.copy(nrm_trn_lbls)
    test_dataset.data = np.concatenate([abn_trn_imgs,nrm_tst_imgs,abn_tst_imgs],axis=0)
    test_dataset.targets = np.concatenate([abn_trn_lbls,nrm_tst_lbls,abn_tst_lbls],axis=0)
    return train_dataset,test_dataset

def _get_cifar_data(train_dataset,test_dataset,user_id):
    '''
    在cifar10数据集下,0-2 3-5 6-9为边端数据 
    其中 0 4 6 作为正常类别
    '''
    train_imgs,train_labels = train_dataset.data,np.array(train_dataset.targets)
    test_imgs,test_labels = test_dataset.data,np.array(test_dataset.targets)    
    
    if user_id == 1:
        
        nrm_trn_idx = np.where(train_labels==0)[0]
        abn_trn_idx = np.concatenate([np.where(train_labels==1)[0],np.where(train_labels==2)[0]])
        nrm_tst_idx = np.where(test_labels==0)[0]
        abn_tst_idx = np.concatenate([np.where(test_labels==1)[0],np.where(test_labels==2)[0]])
        
    elif user_id == 2:

        nrm_trn_idx = np.where(train_labels==4)[0]
        abn_trn_idx = np.concatenate([np.where(train_labels==3)[0],np.where(train_labels==5)[0]])
        nrm_tst_idx = np.where(test_labels==4)[0]
        abn_tst_idx = np.concatenate([np.where(test_labels==3)[0],np.where(test_labels==5)[0]])     

    elif user_id == 3:
        
        nrm_trn_idx = np.where(train_labels==6)[0]
        abn_trn_idx = np.concatenate([np.where(train_labels==7)[0],np.where(train_labels==8)[0],np.where(train_labels==9)[0]])
        nrm_tst_idx = np.where(test_labels==6)[0]
        abn_tst_idx = np.concatenate([np.where(test_labels==7)[0],np.where(test_labels==8)[0],np.where(test_labels==9)[0]])     
        
        
        
    nrm_trn_imgs = train_imgs[nrm_trn_idx]
    nrm_trn_lbls = train_labels[nrm_trn_idx]
    abn_trn_imgs = train_imgs[abn_trn_idx]
    abn_trn_lbls = train_labels[abn_trn_idx]
    nrm_tst_imgs = test_imgs[nrm_tst_idx]
    nrm_tst_lbls = test_labels[nrm_tst_idx]
    abn_tst_imgs = test_imgs[abn_tst_idx]
    abn_tst_lbls = test_labels[abn_tst_idx]

    nrm_trn_lbls[:] = 0
    abn_trn_lbls[:] = 1
    nrm_tst_lbls[:] = 0
    abn_tst_lbls[:] = 1
    
    train_dataset.data = np.copy(nrm_trn_imgs)
    train_dataset.targets = np.copy(nrm_trn_lbls)
    test_dataset.data = np.concatenate([abn_trn_imgs,nrm_tst_imgs,abn_tst_imgs],axis=0)
    test_dataset.targets = np.concatenate([abn_trn_lbls,nrm_tst_lbls,abn_tst_lbls],axis=0)
    return train_dataset,test_dataset

class CIG_Dataset(Dataset):
    def __init__(self,root,split,w=256,h=256):
        super(CIG_Dataset, self).__init__()
        self.root = root
        self.split = split
        self.w = w
        self.h = h

        self.images = []
        self.labels = []
        
        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([w,h]),
            #transforms.Normalize()
        ])
        self.setup()
        
    def setup(self):
        trn_imgs = []
        trn_lbls = []
        tst_imgs = []
        tst_lbls = []
        for i in os.listdir(self.root):
            path = os.path.join(self.root,i)
            if i == 'OK':
                for id in os.listdir(path):
                    path1 = os.path.join(path,id)
                    for j,image in enumerate(os.listdir(path1)):
                        image_path = os.path.join(path1,image)
                        if j < len(os.listdir(path1)) / 3 * 2:
                    
                            trn_imgs.append(image_path)
                            trn_lbls.append(0)
                        else:
                            tst_imgs.append(image_path)
                            tst_lbls.append(0)
            else:
                for id in os.listdir(path):
                    path1 = os.path.join(path,id)
                    for classes in os .listdir(path1):
                        images = os.path.join(path1,classes)
                        for image in os.listdir(images):
                            image_path = os.path.join(images,image)
                            tst_imgs.append(image_path)
                            tst_lbls.append(1)
                            
        if self.split == 'train':
            self.images = trn_imgs
            self.labels = trn_lbls
        elif self.split == 'test':
            self.images = tst_imgs
            self.labels = tst_lbls
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,item):
        image = cv2.imread(self.images[item])
        image = self.transformer(image)
        label = torch.tensor(self.labels[item])
        return image,label