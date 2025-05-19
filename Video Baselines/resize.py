import os
import cv2
import numpy as np
import re
import shutil
import scipy.io
import pandas as pd


def resize_images_in_folder(folder_path, target_size=(128, 128)):
    # 获取文件夹下所有文件列表
    files = os.listdir(folder_path)

    # 遍历所有文件
    for file in files:
        # 构建每个文件的完整路径
        file_path = os.path.join(folder_path, file)

        # 判断文件是否为图片（可以根据文件扩展名进行筛选）
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # 读取图像
            img = cv2.imread(file_path)

            # 如果读取成功，则进行 resize 操作
            if img is not None:
                resized_img = cv2.resize(img, target_size)

                # 保存调整后的图像，覆盖原图像
                cv2.imwrite(file_path, resized_img)
                print(f"Resized {file_path} to {target_size}")
            else:
                print(f"Failed to read {file}")
        else:
            print(f"Skipping non-image file: {file}")


root_path = '/home/jywang/Data/On-Road-rPPG'
for subject_name in sorted(os.listdir(root_path)):
    if not subject_name.startswith('.') and not subject_name.startswith('processed'):
        subject_path_folder = os.path.join(root_path, subject_name)
        for subject_file in os.listdir(subject_path_folder):
            subject_path = os.path.join(subject_path_folder, subject_file)
            align_path = os.path.join(subject_path,'Align')
            print(align_path)
            resize_images_in_folder(align_path)