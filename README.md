## Federated-Memae
联邦学习框架下基于Memae的异常检测架构
## 数据集 
cifar10 mnist
## 模型训练测试命令-cifar10
# cifar10异常检测命令
python main.py --train/test
# 联邦学习框架 Local-training
python main.py --collaborative-training --edge-id 1/2/3 --train/test
# 联邦学习框架 Col-training 
python _main.py --train/test
