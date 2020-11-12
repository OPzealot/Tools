#!/usr/bin/env python
# encoding:utf-8
"""
author: liusili
@contact: liusili@unionbigdata.com
@software:
@file: convert_format
@time: 11/12/2020
@desc: 
"""
import os
import shutil


def convert_img_format(sample_root, img_format=None, tar_format='.jpg'):
    if img_format is None:
        img_format = ['.png', '.JPG']
    for root, _, files in os.walk(sample_root):
        for file in files:
            ori_format = os.path.splitext(file)[1]
            if ori_format in img_format:
                file_path = os.path.join(root, file)
                new_file_path = file_path.replace(ori_format, tar_format)
                shutil.move(file_path, new_file_path)
    print('[FINISH] Convert the format of images.')


if __name__ == '__main__':
    sample_root = r"E:\Working\Visionox\V2_lighter\data\11\1106_test\1106_raw\mura_DMSM1"
    convert_img_format(sample_root)