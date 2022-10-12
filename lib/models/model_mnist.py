import torch
import torch.nn as nn
from options import options
import torch.nn.functional as F

class Minist_Model(nn.Module):
    def __init__(self,cfg):
        super(Minist_Model, self).__init__()
        self.num_memories = cfg.num_memories
        self.feature_size = 16*4*4*4
        self.image_channel_size = 1
        self.conv_channel_size = 16
        self.devices = cfg.devices
        self.num_classes = cfg.num_classes
        self.batch_size = cfg.batch_size
        
        self.encoder = Encoder(image_channel_size=1,conv_channel_size=16)
        self.decoder = Decoder(image_channel_size=1,conv_channel_size=16)
        
        init_mem = torch.zeros(self.num_memories, self.feature_size)
        nn.init.kaiming_uniform_(init_mem)
        self.memory = nn.Parameter(init_mem)
 
        self.cosine_similarity = nn.CosineSimilarity(dim=2,)
        
    def forward(self,x):
        
        batch, channel, height, width = x.size()

        z = self.encoder(x)

        ex_mem = self.memory.unsqueeze(0).repeat(batch, 1, 1)
        ex_z = z.unsqueeze(1).repeat(1, self.num_memories, 1)
        mem_logit = self.cosine_similarity(ex_z, ex_mem)
        mem_weight = F.softmax(mem_logit, dim=1)
        
        z_hat = torch.mm(mem_weight, self.memory) 
        
        logit_x = torch.zeros(self.batch_size,self.num_classes)
        
        rec_x = self.decoder(z_hat)
        
        return dict(rec_x=rec_x,logit_x=logit_x,mem_weight=mem_weight)

class Encoder(nn.Module):
    def __init__(self,image_channel_size,conv_channel_size):
        super(Encoder, self).__init__()
    
        self.image_channel_size = image_channel_size
        self.conv_channel_size = conv_channel_size

        self.conv1 = nn.Conv2d(in_channels=self.image_channel_size,
                               out_channels=self.conv_channel_size,
                               kernel_size=1,
                               stride=2,
                               padding=1,
                              )

        self.bn1 = nn.BatchNorm2d(num_features=self.conv_channel_size,)

        self.conv2 = nn.Conv2d(in_channels=self.conv_channel_size,
                               out_channels=self.conv_channel_size*2,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                              )

        self.bn2 = nn.BatchNorm2d(num_features=self.conv_channel_size*2,)

        self.conv3 = nn.Conv2d(in_channels=self.conv_channel_size*2,
                               out_channels=self.conv_channel_size*4,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                              )

        self.bn3 = nn.BatchNorm2d(num_features=self.conv_channel_size*4,)

        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,x):
        #torch.size([batch_size,1,28,28])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #torch.size([batch_size,16,15,15])
        x = self.conv2(x)
        x = self.bn2(x) 
        x = self.relu(x)
        #torch.size([batch_size,32,8,8])
        x = self.conv3(x) 
        x = self.bn3(x)
        x = self.relu(x)
        #torch.size([batch_size,64,4,4])
        batch, _, _, _ = x.size()
        x = x.view(batch, -1)
        #torch.size([batch_size,1024])
        return x
    

class Decoder(nn.Module):
    def __init__(self,image_channel_size, conv_channel_size):
        super(Decoder, self).__init__()
        self.image_channel_size = image_channel_size
        self.conv_channel_size = conv_channel_size

        self.deconv1 = nn.ConvTranspose2d(in_channels=self.conv_channel_size*4,
                                          out_channels=self.conv_channel_size*2,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          output_padding=1,
                                         )

        self.bn1 = nn.BatchNorm2d(num_features=self.conv_channel_size*2,)

        self.deconv2 = nn.ConvTranspose2d(in_channels=self.conv_channel_size*2,
                                          out_channels=self.conv_channel_size,
                                          kernel_size=2,
                                          stride=2,
                                          padding=1,
                                          output_padding=1,
                                         )

        self.bn2 = nn.BatchNorm2d(num_features=self.conv_channel_size,)

        self.deconv3 = nn.ConvTranspose2d(in_channels=self.conv_channel_size,
                                          out_channels=self.image_channel_size,
                                          kernel_size=2,
                                          stride=2,
                                          padding=1,
                                         )

        self.relu = nn.ReLU(inplace=True)
        
        
        
    def forward(self,x):
        x = x.view(-1, self.conv_channel_size*4, 4, 4)

        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.deconv3(x)
        return x