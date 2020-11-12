#!/usr/bin/env python
# encoding:utf-8
"""
author: liusili
@contact: liusili@unionbigdata.com
@software:
@file: check_dataset
@time: 8/25/2020
@desc: 
"""
import cv2
import os
import sys
from tqdm import tqdm
import xml.etree.ElementTree as ET


def check_valid_image(sample_root, img_format='.jpg'):
    for root, _, file_lst in os.walk(sample_root):
        pbar = tqdm(file_lst, file=sys.stdout)
        for file in pbar:
            if os.path.splitext(file)[-1] == img_format:
                category = root.split('\\')[-1]
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                try:
                    img.shape
                except Exception as e:
                    tqdm.write('[ERROR]:{}, {}'.format(category, file))
                pbar.set_description('Processing [{}]'.format(category))
    print('Finish checking images.')


def get_and_check(root, name, length):
    """
    :param root: Element-tree 根节点
    :param name: 需要返回的子节点名称
    :param length: 确认子节点长度
    """
    var_lst = root.findall(name)
    if len(var_lst) == 0:
        raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
    if (length > 0) and (len(var_lst) != length):
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'
                                  % (name, length, len(var_lst)))
    if length == 1:
        var_lst = var_lst[0]
    return var_lst


def find_label(sample_root, label, replace=None):
    for rootPath, _, file_lst in os.walk(sample_root):
        pbar = tqdm(file_lst, file=sys.stdout)
        for file in pbar:
            if os.path.splitext(file)[-1] == '.xml':
                category = rootPath.split('\\')[-1]
                xml_path = os.path.join(rootPath, file)
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for obj in root.findall('object'):
                    name = get_and_check(obj, 'name', 1).text
                    if name == label:
                        tqdm.write('[FIND]:{}, {}'.format(category, file))
                        if replace:
                            get_and_check(obj, 'name', 1).text = replace
                            tree.write(xml_path)
                            tqdm.write('[REPLACE] {} ==> {}'.format(label, replace))
                pbar.set_description('Processing [{}]'.format(category))
    print('Finish finding process.')


if __name__ == '__main__':
    sample_root = r"E:\Working\Visionox\V2_lighter\data\11\lighter_1105"
    find_label(sample_root, 'DMSM1_K1', replace='DMSM1')