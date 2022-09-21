'''
Author: jyniki 1067087283@qq.com
Date: 2022-05-19 16:49:01
LastEditors: jyniki 1067087283@qq.com
LastEditTime: 2022-09-21 14:31:47
FilePath: /new_memae/main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
from torchvision import datasets
from torchvision import transforms
from lib.data.dataset import get_dataset,get_cifar_anomaly_dataset,get_mnist_anomaly_dataset,Data,get_cifar_data,_get_cifar_data
from lib.models import model_cifar,model_mnist,model_MVTec,model_CIG
from options import options
from torch.utils.data import DataLoader
from utils import set_workdir,logger,train_one_epoch,test_one_epoch
from tqdm import tqdm
from torch.optim import Adam
import numpy as np
import torch.nn as nn
import os

def main(opt):
 
    if opt.data_type == 'cifar10':
        train_dataset,test_dataset = get_dataset(opt)
        train_dataset,test_dataset = get_cifar_anomaly_dataset(train_dataset,test_dataset,nrm_cls_idx=train_dataset.class_to_idx[opt.nrm_class])
        model = model_cifar.CIFAR_Model(opt)
    elif opt.data_type == 'mnist':
        train_dataset,test_dataset = get_dataset(opt)
        train_dataset,test_dataset = get_mnist_anomaly_dataset(train_dataset,test_dataset,nrm_cls_idx=int(opt.nrm_class))
        model = model_mnist.Minist_Model(opt)
    elif opt.data_type == 'MVTec':
        train_dataset,test_dataset = get_dataset(opt)
    elif opt.data_type == 'CIG':
        opt.nrm_class = 'cigar'
        train_dataset,test_dataset = get_dataset(opt)
        model = model_CIG.CIG_Model(opt)
    print('数据类型:{}'.format(opt.data_type), '训练集包含{}张'.format(len(train_dataset)), '测试集包含{}张'.format(len(test_dataset)))
    
    ########################
    ###边端的单独训练模式#####
    
    if opt.collaborative_training:
        train_dataset,test_dataset = get_dataset(opt)
        train_dataset,test_dataset = _get_cifar_data(train_dataset,test_dataset,user_id=opt.edge_id) ###在使用不同数量边端的时候这一块需要做改变
        model = model_cifar.CIFAR_Model(opt)
        print('Local training','边端id{}'.format(opt.edge_id),'训练集包含{}张'.format(len(train_dataset)),'测试集包含{}张'.format(len(test_dataset)))
        
    if torch.cuda.device_count() > 0 and opt.multi_gpu:
        model = nn.DataParallel(model)
        
    model = model.to(opt.devices)
    optimizer = Adam(model.parameters(),lr=opt.lr)
    train_dataloader = DataLoader(train_dataset,batch_size=opt.batch_size)
    test_dataloader = DataLoader(test_dataset,batch_size=opt.batch_size)
    data = Data(train_dataloader,test_dataloader)
    set_workdir(opt.work_dir)
    if opt.collaborative_training:
        path = os.path.join(opt.work_dir,'logs/fed{}_{}_{}_{}_log.txt'.format(opt.edge_id,opt.data_type,opt.addressing,opt.nrm_class))
        PATH = os.path.join(opt.work_dir,'ckpt/{}/fed{}_{}_{}_state_dict_model.pth'.format(opt.data_type,opt.edge_id,opt.nrm_class,opt.addressing))
    else:
        path = os.path.join(opt.work_dir,'logs/{}_{}_{}_log.txt'.format(opt.data_type,opt.addressing,opt.nrm_class))
        PATH = os.path.join(opt.work_dir,'ckpt/{}/{}_{}_state_dict_model.pth'.format(opt.data_type,opt.nrm_class,opt.addressing))
    if opt.train:
        if os.path.exists(path):
            os.remove(path)
        logger(path,'training for detecting {}'.format(opt.nrm_class))        
        best_roc_auc = 0
        for epoch in range(opt.num_epochs):
            logger(logger_path=path,logger_info='{}/{}'.format(epoch+1,opt.num_epochs))
            loss = train_one_epoch(opt, data, model, optimizer)    
            logger(logger_path=path,logger_info='loss:{}'.format(loss))
            
            
            
            if (epoch + 1) % 10 == 0:
                roc_auc = test_one_epoch(opt,data,model)
                if best_roc_auc < roc_auc:
                    best_roc_auc = roc_auc
                    
                    torch.save(model.state_dict(),PATH)
                logger(logger_path=path,logger_info='roc_auc:{}  best_roc_auc:{}'.format(roc_auc,best_roc_auc))
    elif opt.test:
        PATH = os.path.join(opt.work_dir,'ckpt/{}/{}_{}_state_dict_model.pth'.format(opt.data_type,opt.nrm_class,opt.addressing))
        if opt.collaborative_training:
            PATH = os.path.join(opt.work_dir,'ckpt/{}/fed{}_{}_{}_state_dict_model.pth'.format(opt.data_type,opt.edge_id,opt.nrm_class,opt.addressing))
        model.load_state_dict(torch.load(PATH))
        roc_auc = test_one_epoch(opt, data, model)
        print('roc_auc:{}'.format(roc_auc))
        
    
    
    
if __name__ == '__main__':
    opt = options()
    main(opt)