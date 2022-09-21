'''
Author: jyniki 1067087283@qq.com
Date: 2022-05-23 19:00:05
LastEditors: jyniki 1067087283@qq.com
LastEditTime: 2022-05-27 12:02:06
FilePath: /new_memae/_main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
from torch.utils.data import DataLoader
from lib.data.dataset import get_dataset,get_cifar_anomaly_dataset,get_mnist_anomaly_dataset,Data
from lib.models import model_cifar,model_mnist,model_MVTec
from lib.models.model_cifar import CIFAR_Model
from lib.data.dataset import Data,get_dataset,get_cifar_data,_get_cifar_data
import os
import torch.nn as nn
from torch.optim import Adam
from utils import train_one_epoch,test_one_epoch,logger,fed_avg,para_avg
from options import options

def main(opt):
    # model    
    model1 = CIFAR_Model(opt)
    model2 = CIFAR_Model(opt)
    model3 = CIFAR_Model(opt)
    model1 = model1.to(opt.devices)
    model2 = model2.to(opt.devices)
    model3 = model3.to(opt.devices)
    # data1
    train_dataset,test_dataset = get_dataset(opt)
    train_dataset_1,test_dataset_1 = _get_cifar_data(train_dataset,test_dataset,1)
    #print(len(train_dataset_1),len(test_dataset_1))
    
    train_dataloader_1 = DataLoader(train_dataset_1,batch_size=opt.batch_size)
    test_dataloader_1 = DataLoader(test_dataset_1,batch_size=opt.batch_size)
    data1 = Data(train_dataloader_1,test_dataloader_1)
    # data2
    train_dataset,test_dataset = get_dataset(opt)
    train_dataset_2,test_dataset_2 = _get_cifar_data(train_dataset,test_dataset,2)    
    #print(len(train_dataset_2), len(test_dataset_2))
    
    train_dataloader_2 = DataLoader(train_dataset_2,batch_size=opt.batch_size)
    test_dataloader_2 = DataLoader(test_dataset_2,batch_size=opt.batch_size)
    data2 = Data(train_dataloader_2,test_dataloader_2)    
    #data3
    train_dataset,test_dataset = get_dataset(opt)
    train_dataset_3,test_dataset_3 = _get_cifar_data(train_dataset,test_dataset,3)
    
    train_dataloader_3 = DataLoader(train_dataset_3,batch_size=opt.batch_size)
    test_dataloader_3 = DataLoader(test_dataset_3,batch_size=opt.batch_size)
    data3 = Data(train_dataloader_3,test_dataloader_3)   
    
    optimizer1 = Adam(model1.parameters(),lr=opt.lr)
    optimizer2 = Adam(model2.parameters(),lr=opt.lr)
    optimizer3 = Adam(model3.parameters(),lr=opt.lr)
    
    path = os.path.join(opt.work_dir,'logs/edge{}_{}_{}_log.txt'.format(opt.edge_id,opt.data_type,opt.addressing))
    if os.path.exists(path):
        os.remove(path)
        
    best_roc_auc1 = 0
    best_roc_auc2 = 0
    best_roc_auc3 = 0
    _sum = 0
    
    if opt.test:
        PATH = os.path.join(opt.work_dir,'ckpt/{}/edge{}_{}_state_dict_model.pth'.format(opt.data_type,opt.edge_id,opt.addressing))
        model1.load_state_dict(torch.load(PATH))
        roc_auc1 = test_one_epoch(opt,data1,model1)
        roc_auc2 = test_one_epoch(opt,data2,model1)
        roc_auc3 = test_one_epoch(opt,data3,model1)
        print('roc_auc 1 : {}  roc_auc 2 : {}  roc_auc3 : {} '.format(roc_auc1,roc_auc2,roc_auc3))
        
    if opt.train:
        for epoch in range(opt.num_epochs):
            logger(logger_path=path,logger_info='{}/{}'.format(epoch+1,opt.num_epochs))
            loss1 = train_one_epoch(opt, data1, model1, optimizer1)
            loss2 = train_one_epoch(opt, data2, model2, optimizer2) 
            loss3 = train_one_epoch(opt, data3, model3, optimizer3)
            logger(logger_path=path,logger_info='loss1:{} loss2:{} loss:{}'.format(loss1,loss2,loss3))
            model = nn.ModuleList([model1, model2, model3])
            avg_dict = para_avg(model)
            model1.load_state_dict(avg_dict)
            model2.load_state_dict(avg_dict)
            model3.load_state_dict(avg_dict)
            roc_auc1 = test_one_epoch(opt, data1, model1)
            roc_auc2 = test_one_epoch(opt, data2, model2)
            roc_auc3 = test_one_epoch(opt, data3, model3)
            logger(logger_path=path,logger_info='roc_auc1:{} roc_auc2:{} roc_auc3:{}'.format(roc_auc1,roc_auc2,roc_auc3))
        
            if best_roc_auc1 < roc_auc1:
                best_roc_auc1 = roc_auc1
            if best_roc_auc2 < roc_auc2:    
                best_roc_auc2 = roc_auc2
            if best_roc_auc3 < roc_auc3:
                best_roc_auc3 = roc_auc3
            logger(logger_path=path,logger_info='best_roc_auc1:{} best_roc_auc2:{} best_roc_auc3:{}'.format(best_roc_auc1,best_roc_auc2,best_roc_auc3))
        
            if best_roc_auc1 + best_roc_auc2 + best_roc_auc3 > _sum:
                _sum = best_roc_auc1 + best_roc_auc2 + best_roc_auc3
                PATH = os.path.join(opt.work_dir,'ckpt/{}/edge{}_{}_state_dict_model.pth'.format(opt.data_type,opt.edge_id,opt.addressing))
                torch.save(model1.state_dict(),PATH)
            
if __name__ == '__main__':
    opt = options()
    main(opt)