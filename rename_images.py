#!/usr/bin/env python
# encoding:utf-8
"""
author: liusili
@contact: liusili@unionbigdata.com
@software:
@file: rename_images
@time: 2020/6/28
@desc: 
"""
import os
import shutil
from datetime import datetime


def convert_img_format(sample_root, img_format=None, tar_format='.jpg'):
    if img_format is None:
        img_format = ['.jpg', '.JPG', '.jfif']
    for root, _, files in os.walk(sample_root):
        for file in files:
            ori_format = os.path.splitext(file)[1]
            if ori_format in img_format:
                file_path = os.path.join(root, file)
                new_file_path = file_path.replace(ori_format, tar_format)
                shutil.move(file_path, new_file_path)
    print('[FINISH] Convert the format of images.')


def rename_images(sample_root, img_name, img_format=('.jpg', '.JPG'), digit=4):
    i = 1
    now = datetime.now()
    # date = '{:0>2d}{:0>2d}'.format(now.month, now.day)
    date = '{:0>2d}{:0>2d}'.format(11, 6)

    for root, _, file_lst in os.walk(sample_root):
        for file in file_lst:
            if os.path.splitext(file)[-1] in img_format:
                file_name = os.path.splitext(file)[0]
                img_path = os.path.join(root, file)
                new_name = '{}_{}_{:0>{}d}'.format(img_name, date, i, digit)
                new_img_path = os.path.join(root, new_name + '.jpg')
                shutil.move(img_path, new_img_path)

                xml_path = os.path.join(root, file_name + '.xml')
                if os.path.isfile(xml_path):
                    new_xml_path = os.path.join(root, new_name + '.xml')
                    shutil.move(xml_path, new_xml_path)
                i += 1

    print('finish.')


if __name__ == '__main__':
    sample_root = r"E:\Working\Visionox\V2_lighter\data\11\1106\1106_raw"
    img_name = 'lighter'
    # convert_img_format(sample_root)
    rename_images(sample_root, img_name, digit=2)
