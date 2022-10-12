'''
Author: jyniki 1067087283@qq.com
Date: 2022-05-30 16:22:48
LastEditors: jyniki 1067087283@qq.com
LastEditTime: 2022-05-30 20:25:52
FilePath: /new_memae/lib/data/prepocess.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
root = '/data0/JY/xxz/Anomaly/data/利群'

images_list_0 = []
images_list_1 = []
images_list_2 = []
images_list_3 = []

for first_path in os.listdir(root):
    if first_path == 'NG':
        first_path = os.path.join(root, first_path)
        for second_path in os.listdir(first_path):
            num_id = second_path
            second_path = os.path.join(first_path, second_path)
            for third_path in os.listdir(second_path):
                third_path = os.path.join(second_path, third_path)
                for image_path in os.listdir(third_path):
                    image_path = os.path.join(third_path, image_path)
     
                    if num_id == '0':
                        images_list_0.append(image_path)
                    elif num_id == '1':
                        images_list_1.append(image_path)
                    elif num_id == '2':
                        images_list_2.append(image_path)
                    elif num_id == '3':
                        images_list_3.append(image_path)                   
    elif first_path == 'OK':
        first_path = os.path.join(root, first_path)
        for second_path in os.listdir(first_path):
            num_id = second_path
            second_path = os.path.join(first_path, second_path)
            for image_path in os.listdir(second_path):
                image_path = os.path.join(second_path, image_path)
                if num_id == '0':
                    images_list_0.append(image_path)
                elif num_id == '1':
                    images_list_1.append(image_path)
                elif num_id == '2':
                    images_list_2.append(image_path)
                elif num_id == '3':
                    images_list_3.append(image_path)

crop_box_0 = [98,30,765,395]
crop_box_1 = [70,38,740,411]
crop_box_2 = [98,229,1280,411]
crop_box_3 = [210,230,1050,750]
for image_path in images_list_0:
    image = Image.open(image_path)
    image_crop = image.crop(crop_box_0)
    image_path = image_path.replace('利群', 'cropped_image')
    image_crop.save(image_path)

for image_path in images_list_1:
    image = Image.open(image_path)
    image_crop = image.crop(crop_box_1)
    image_path = image_path.replace('利群', 'cropped_image')
    image_crop.save(image_path)

for image_path in images_list_2:
    image = Image.open(image_path)
    image_crop = image.crop(crop_box_2)
    image_path = image_path.replace('利群', 'cropped_image')
    image_crop.save(image_path)


for image_path in images_list_3:
    image = Image.open(image_path)
    image_crop = image.crop(crop_box_3)
    image_path = image_path.replace('利群', 'cropped_image')
    image_crop.save(image_path)
