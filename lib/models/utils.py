'''
Author: jyniki 1067087283@qq.com
Date: 2022-05-18 15:59:35
LastEditors: jyniki 1067087283@qq.com
LastEditTime: 2022-05-19 13:50:35
FilePath: /new_memae/lib/models/utils.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import importlib

def load_model(opt,dataloader):
    '''
    load model based on the model name
    '''
    
    model_name = opt.model
    model_path = f'lib.models.{model_name}'
    model_lib = importlib.import_module(model_path)
    model = getattr(model_lib,model_name.title())   #title属性返回标题化的字符串
    return model(opt,dataloader)