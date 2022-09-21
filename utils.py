'''
Author: jyniki 1067087283@qq.com
Date: 2022-05-19 14:23:38
LastEditors: jyniki 1067087283@qq.com
LastEditTime: 2022-05-31 11:11:21
FilePath: /new_memae/utils.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score,roc_auc_score
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch
from lib.models.model_cifar import CIFAR_Model
import collections
def train_one_epoch(opt,data,model,optimizer):
    model.train()
    for images,labels in tqdm(data.train):
        images = images.to(opt.devices)
        outputs = model(images)
        rec_x = outputs['rec_x']
        logit_x = outputs['logit_x']
        mem_weight = outputs['mem_weight']
        rec_loss = l2_loss(rec_x,images)
        rec_loss = torch.sum(rec_loss)/opt.batch_size
        if opt.addressing == 'sparse':
            mask = (mem_weight == 0).float()
            mask_weight = mem_weight + mask
            entropy_loss = -mem_weight * torch.log(mask_weight)
            entropy_loss = entropy_loss.sum() / opt.batch_size
            entropy_loss *= opt.entropy_loss_coef
        elif opt.addressing == 'soft':
            entropy_loss = torch.zeros(1).to(opt.devices)
        loss = rec_loss + entropy_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss
        
def test_one_epoch(opt,data,model):
    model.eval()
    labels_list = []
    anomaly_score = []
    rec_criterion = nn.MSELoss(reduction='none')
    with torch.no_grad():
        for images,labels in tqdm(data.test):
            print(images.shape)
            images = images.to(opt.devices)
            outputs = model(images)
            rec_x = outputs['rec_x']
            logit_x = outputs['logit_x']
            mem_weight = outputs['mem_weight']
            
            rec_loss = rec_criterion(rec_x,images)
            rec_loss = torch.sum(rec_loss,axis=[1,2,3])
            rec_loss = rec_loss.detach().cpu().numpy()
            
            labels_list.extend(labels) 
            anomaly_score.extend(rec_loss)
        roc_auc = get_anomaly_score(anomaly_score,labels_list)
        return roc_auc 
                  
def set_workdir(workdir):   #'./workdir'
    if not os.path.exists(workdir):
        os.makedirs(workdir)
    if not os.path.exists(os.path.join(workdir,'ckpt')):
        os.makedirs(os.path.join(workdir,'ckpt'))
    if not os.path.exists(os.path.join(workdir,'logs')):
        os.makedirs(os.path.join(workdir,'logs'))

def logger(logger_path,logger_info):
    with open(logger_path,'a') as f:
        f.write(logger_info + '\n')
    print(logger_info)

def l1_loss(input,target):
    #这里返回的应该是一个 数值
    '''
    L1 Loss without reduce flag
    
    Args:
        input(FloatTensor):Input tensor
        target(FloatTensor):Output tensor
        
    Returns:
        [FloatTensor]:L1 distance between input and target
    '''
    return torch.mean(torch.abs(input - target))

def l2_loss(input,target,size_average=True):
    #这里返回的不一定是一个数值
    '''
    L2 Loss without reduce flag
    
    Args:
        input(FloatTensor):Input tensor
        target(FloatTensor):Output tensor
        
    Returns:
        [FloatTensor]:L1 distance between input and target
    '''
    if size_average:    
        #这里没用采用平均值 结果是一样的 为了能让损失更大 更加的显示化
        return torch.sum(torch.pow((input-target),2))
    else:
        return torch.pow((input-target),2)
    
def get_anomaly_score(anomaly_score,labels_list):
   anomaly_score = np.array(anomaly_score)
   labels_list = np.array(labels_list)
   anomaly_score = (anomaly_score - np.min(anomaly_score)) / (np.max(anomaly_score) - np.min(anomaly_score))
   roc_auc = roc_auc_score(labels_list, anomaly_score)
   #print('test conclusion for detecting {} : \n{}%'.format(opt.nrm_class,roc_auc*100))
   return roc_auc

def fed_avg(opt,model1,model2):
    model3 = CIFAR_Model(opt)
    model3 = model3.to(opt.devices)
    for name1,param1 in model1.named_parameters():
        for name2,param2 in model2.named_parameters():
            for name3,param3 in model3.named_parameters():
                if name1 == name2:
                    if name2 == name3:
                        param3.data = (param1.data + param2.data) / 2
    return model3

def para_avg(model_pre):
    """
    这是模型参数平均函数！值得记忆！
    输入是多个模型组成的ModuleList输出是平均化后的模型参数，目标模型声明以后再使用load_state_dict()函数导入平均的参数即可！
    """
    worker_state_dict = [x.state_dict() for x in model_pre]
    weight_keys = list(worker_state_dict[0].keys())
    fed_state_dict = collections.OrderedDict()
    for key in weight_keys:
        key_sum = 0
        for i in range(len(model_pre)):
            key_sum = key_sum + worker_state_dict[i][key]
        fed_state_dict[key] = key_sum / len(model_pre)

    return fed_state_dict