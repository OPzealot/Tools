#!/usr/bin/env python
# encoding:utf-8
"""
author: liusili
@contact: liusili@unionbigdata.com
@software:
@file: defect_generator
@time: 8/13/2020
@desc: 
"""
import os
import cv2
import random
from tqdm import tqdm
from glob import glob
import numpy as np
import albumentations as A
from generate_xml import generate_xml


def get_defect(defect_path, shape):
    img = cv2.imread(defect_path)
    img = cv2.resize(img, shape, interpolation=cv2.INTER_LINEAR)
    img = A.ShiftScaleRotate(shift_limit=0.1, scale_limit=(0, 0.2),
                             rotate_limit=180, always_apply=True)(image=img)['image']
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = np.ones((7, 7), np.uint8)
    gray = cv2.dilate(gray, kernel, iterations=2)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bbox_lst = []
    for c in contours:
        bbox = cv2.cv2.boundingRect(c)
        bbox_lst.append(bbox)
        # x, y, w, h = bbox
        # img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img, bbox_lst


def merge_defect(src, defect):
    augment = A.Compose([
        A.MedianBlur(blur_limit=5, always_apply=True),
        A.RandomBrightnessContrast(brightness_limit=(-0.5, -0.6),
                                   contrast_limit=(-0.4, -0.3), always_apply=True),
    ], p=1)
    defect = augment(image=defect)['image']
    height, width, _ = src.shape
    center = (width // 2, height // 2)
    mask = 255 * np.ones(src.shape, src.dtype)
    mixed = cv2.seamlessClone(defect, src, mask, center, cv2.MIXED_CLONE)
    return mixed


def generate_defect(sample_root, defect_root, defect_name):
    defect_lst = os.listdir(defect_root)
    out_path = sample_root.rstrip(sample_root.split('\\')[-1]) + defect_name
    os.makedirs(out_path, exist_ok=True)
    pbar = tqdm(glob(sample_root + '/*/*.jpg'))
    i = 200
    for src_path in pbar:
        defect_path = os.path.join(defect_root, random.choice(defect_lst))
        src = cv2.imread(src_path)
        h, w, _ = src.shape
        defect, bbox_lst = get_defect(defect_path, shape=(w, h))
        mixed = merge_defect(src, defect)
        i += 1
        mixed_name = '{}_{:0>{}d}'.format(defect_name, i, 3)

        img_path = os.path.join(out_path, mixed_name + '.jpg')
        xml_path = os.path.join(out_path, mixed_name + '.xml')

        img_info = {'file_name': mixed_name, 'path': img_path,
                    'height': h, 'width': w}
        generate_xml(img_info, defect_name, bbox_lst)
        cv2.imwrite(img_path, mixed)
        pbar.set_description('Generating defect images')

    print('Finish.')


if __name__ == '__main__':
    src_root = r'C:\Users\OPzealot\Desktop\FALSE'
    crack_root = r'C:\Users\OPzealot\Desktop\crack'
    defect = 'AZ01'
    generate_defect(src_root, crack_root, defect)