'''
Author: jyniki 1067087283@qq.com
Date: 2022-05-18 16:19:04
LastEditors: jyniki 1067087283@qq.com
LastEditTime: 2022-05-27 15:33:42
FilePath: /new_memae/options.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import argparse

'''
#在切换2个边端和3个边端的时候需要 需要分别改变main和_main当中获取dataset和user_id部分
cifar10 数据集
2个边端的情况下 : 在边端单独训练 python main.py --collaborative-training --edge-id 1/2 --train/test 
                 在云端联合训练 python _main.py --train/test 
3个边端的情况下 : 在边端单独训练 python main.py --collaborative-training --edge-id 1/2/3 --train/test 
                 在云端联合训练 python _main.py --train/test
                 
使用烟包数据集
在边端单独训练 
'''

def options():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data-type',type=str,default='cifar10')
    parser.add_argument('--data-root',type=str,default='../data')
    parser.add_argument('--nrm-class',type=str,default='airplane')
    parser.add_argument('--devices',type=str,default='cuda')
    parser.add_argument('--addressing',type=str,default='sparse')
    parser.add_argument('--work-dir',type=str,default='./workdir')
    
    parser.add_argument('--train',action='store_true',default=False)
    parser.add_argument('--test',action='store_true',default=False)
    parser.add_argument('--visualize',action='store_true',default=False)
    parser.add_argument('--multi-gpu',action='store_true',default=False)
    parser.add_argument('--collaborative-training',action='store_true',default=False)
    
    parser.add_argument('--batch-size',type=int,default=512)
    parser.add_argument('--num-epochs',type=int,default=300)
    parser.add_argument('--num-memories',type=int,default=100)
    parser.add_argument('--conv-channel-size',type=int,default=64)  #cifar 64 mnist 16
    parser.add_argument('--image-channel-size',type=int,default=3)
    parser.add_argument('--num-classes',type=int,default=10)
    parser.add_argument('--edge-id',type=int,default=0)
    
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--entropy_loss_coef',type=float,default=0.0002)
    opt = parser.parse_args()
    return opt