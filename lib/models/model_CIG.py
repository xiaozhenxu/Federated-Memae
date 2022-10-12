import torch
import torch.nn as nn
from options import options
import torch.nn.functional as F

class CIG_Model(nn.Module):
    '''
    (3,256,256) -> (64,128,128) -> (128,64,64) -> (128,32,32) -> (256,16,16) -> (256,8,8) -> (512,4,4)
    '''
    def __init__(self,cfg):
        super(CIG_Model, self).__init__()
        self.num_memories = cfg.num_memories
        self.feature_size = cfg.conv_channel_size*4*4
        self.image_channel_size = cfg.image_channel_size
        self.conv_channel_size = cfg.conv_channel_size
        self.devices = cfg.devices     
        self.num_classes = cfg.num_classes
        self.batch_size = cfg.batch_size
        self.cfg = cfg
        self.encoder = Encoder(self.image_channel_size,self.conv_channel_size)
        self.decoder = Decoder(self.image_channel_size,self.conv_channel_size)
        self.addressing = cfg.addressing


        init_mem = torch.zeros(self.num_memories,512*4*4)
        nn.init.kaiming_uniform_(init_mem)
        self.memory = nn.Parameter(init_mem)
        
        self.cosine_similarity = nn.CosineSimilarity(dim=2)  #dim=2 如果这里dim不等于2就会出现报错 dim等于多少就相当于哪个维度会损失 ，相当于在外循环内部又套上内循环
        self.relu = nn.ReLU(inplace=True)
        if self.addressing == 'sparse':
            self.threshold = 1/self.memory.size(0)
            self.epsilon = 1e-15
    
    def train_one_epoch(self,train_dataloader,optimizer):
        pass
    
    def test_one_epoch(self,test_dataloader):
        pass
        
    def logger(self,logger_path,logger_info):
        with open(logger_path,'a') as f:
            f.write(logger_info + '\n')
        print(logger_info)
    

    def forward(self,x):
        batch,channel,height,width = x.size()
        z = self.encoder(x)

        #生成参数m
        ex_mem = self.memory.unsqueeze(0).repeat(batch, 1, 1)
        ex_z = z.unsqueeze(1).repeat(1, self.num_memories, 1)

        mem_logit = self.cosine_similarity(ex_z, ex_mem)

        mem_weight = F.softmax(mem_logit, dim=1)
        
        #这一块之后需要改动
        #没对m做矩阵稀疏化 直接利用m生成z_hat 
        #如果想对矩阵做稀疏化处理 这里还有一项正则化损失函数
        if self.addressing == 'soft':
            z_hat = torch.mm(mem_weight,self.memory)
        elif self.addressing == 'sparse':
            mem_weight = (self.relu(mem_weight-self.threshold)*mem_weight)/(torch.abs(mem_weight-self.threshold)+self.epsilon)
            mem_weight = mem_weight / mem_weight.norm(p=1, dim=1).unsqueeze(1).expand(batch, self.num_memories)
            z_hat = torch.mm(mem_weight,self.memory)
            
        #这里有一项logit输出 不太清楚是什么意思 
        #因为这里也不想加入logit损失函数 所以直接赋值为0
        logit_x = torch.zeros(self.batch_size,self.num_classes)
        
        rec_x = self.decoder(z_hat)
        
        return dict(rec_x=rec_x,logit_x=logit_x,mem_weight=mem_weight)
    
class Encoder(nn.Module):
    def __init__(self,image_channel_size,conv_channel_size):
        super(Encoder, self).__init__()
        self.image_channel_size = image_channel_size
        self.conv_channel_size = conv_channel_size
        self.conv1 = nn.Conv2d(
                                in_channels=self.image_channel_size,
                                out_channels=self.conv_channel_size,
                                kernel_size=3,
                                stride=2,
                                padding=1
                               )
        
        self.bn1 = nn.BatchNorm2d(num_features=self.conv_channel_size)
        
        self.conv2 = nn.Conv2d(
                                in_channels=self.conv_channel_size,
                                out_channels=self.conv_channel_size*2,
                                kernel_size=3,
                                stride=2,
                                padding=1
        )
        
        self.bn2 = nn.BatchNorm2d(num_features=self.conv_channel_size*2)

        self.conv3 = nn.Conv2d(
                                in_channels=self.conv_channel_size*2,
                                out_channels=self.conv_channel_size*2,
                                kernel_size=3,
                                stride=2,
                                padding=1
        )
        
        self.bn3 = nn.BatchNorm2d(num_features=self.conv_channel_size*2)
        
        self.conv4 = nn.Conv2d(
                                in_channels=self.conv_channel_size*2,
                                out_channels=self.conv_channel_size*2*2,
                                kernel_size=3,
                                stride=2,
                                padding=1
        )
        self.bn4 = nn.BatchNorm2d(num_features=self.conv_channel_size*2*2)
        
        self.conv5 = nn.Conv2d(
                                in_channels=self.conv_channel_size*2*2,
                                out_channels=self.conv_channel_size*2*2,
                                kernel_size = 3,
                                stride = 2,
                                padding = 1
        )
        self.bn5 = nn.BatchNorm2d(num_features=self.conv_channel_size*2*2)
        
        self.conv6 = nn.Conv2d(
                                in_channels = self.conv_channel_size*2*2,
                                out_channels = self.conv_channel_size*2*2*2,
                                kernel_size = 3,
                                stride = 2,
                                padding = 1,
        )
        self.bn6 = nn.BatchNorm2d(num_features=self.conv_channel_size*2*2*2)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)    # 512*7*7
        
        batch_size,_,_,_ = x.size()
        x = x.view(batch_size,-1)
        return x
    
class Decoder(nn.Module):
    def __init__(self,image_channel_size,conv_channel_size):
        super(Decoder,self).__init__()
        self.image_channel_size = image_channel_size
        self.conv_channel_size = conv_channel_size
        
        self.dconv1 = nn.ConvTranspose2d(
            in_channels = self.conv_channel_size*2*2*2,
            out_channels = self.conv_channel_size*2*2,
            kernel_size = 3,
            stride = 2,
            padding = 1,
            output_padding = 1
        )
        
        self.bn1 = nn.BatchNorm2d(num_features=self.conv_channel_size*2*2)
        
        self.dconv2 = nn.ConvTranspose2d(
            in_channels = self.conv_channel_size*2*2,
            out_channels = self.conv_channel_size*2*2,
            kernel_size = 3,
            stride = 2,
            padding = 1,
            output_padding=1
        )
        
        self.bn2 = nn.BatchNorm2d(num_features=self.conv_channel_size*2*2)
        
        self.dconv3 = nn.ConvTranspose2d(
            in_channels = self.conv_channel_size*2*2,
            out_channels = self.conv_channel_size*2,
            kernel_size = 3,
            stride = 2,
            padding = 1,
            output_padding=1
        )
        
        self.bn3 = nn.BatchNorm2d(num_features=self.conv_channel_size*2)
        
        self.dconv4 = nn.ConvTranspose2d(
            in_channels = self.conv_channel_size*2,
            out_channels = self.conv_channel_size*2,
            kernel_size = 3,
            stride = 2,
            padding = 1,
            output_padding=1
        )
        self.bn4 = nn.BatchNorm2d(num_features=self.conv_channel_size*2)
        
        self.dconv5 = nn.ConvTranspose2d(
            in_channels = self.conv_channel_size*2,
            out_channels = self.conv_channel_size,
            kernel_size = 3,
            stride = 2,
            padding = 1,
            output_padding=1
        )
        self.bn5 = nn.BatchNorm2d(num_features=self.conv_channel_size)
        
        self.dconv6 = nn.ConvTranspose2d(
            in_channels = self.conv_channel_size,
            out_channels = self.image_channel_size,
            kernel_size = 3,
            stride = 2,
            padding = 1,
            output_padding = 1
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self,x):
        batch_size,_ = x.size()
        

        x = x.view(batch_size,self.conv_channel_size*8,7,7)
        
        #x = x.view(batch_size,self.conv_channel_size*4,16,16)
        x = self.dconv1(x)
        x = self.bn1(x) 
        x = self.relu(x)
        
        x = self.dconv2(x)
        x = self.bn2(x)
        x = self.relu(x)   
        
        x = self.dconv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.dconv4(x)

        x = self.bn4(x)
        x = self.relu(x)
        
        x = self.dconv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        
        x = self.dconv6(x)
        
        return x
    
