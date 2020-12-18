#!/usr/bin/env python
# encoding:utf-8
"""
author: liusili
@contact: liusili@unionbigdata.com
@software:
@file: mask_defect_generator
@time: 8/25/2020
@desc: 
"""
import os
import cv2
import random
import xml.etree.ElementTree as ET


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


def extract_bbox(xml_path):
    bbox_lst = []
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for obj in root.findall('object'):
        bbox = get_and_check(obj, 'bndbox', 1)
        xmin = int(get_and_check(bbox, 'xmin', 1).text)
        ymin = int(get_and_check(bbox, 'ymin', 1).text)
        xmax = int(get_and_check(bbox, 'xmax', 1).text)
        ymax = int(get_and_check(bbox, 'ymax', 1).text)

        if xmin < 5 or ymin < 5 or xmax > 250 or ymax > 250:
            continue

        bbox_lst.append((xmin, ymin, xmax, ymax))
    return bbox_lst


def get_random_pair(bbox_lst):
    bbox = bbox_lst[0]
    bboxWidth = bbox[2] - bbox[0]
    bboxHeight = bbox[3] - bbox[1]
    for i in range(20):
        bbox1, bbox2 = random.choices(bbox_lst, k=2)
        center1 = (bbox1[0]+bbox1[2])/2, (bbox1[1]+bbox1[3])/2
        center2 = (bbox2[0]+bbox2[2])/2, (bbox2[1]+bbox2[3])/2
        xGap = abs(center1[0] - center2[0])
        yGap = abs(center1[1] - center2[1])
        if xGap < 2*bboxWidth and yGap < 2.5*bboxHeight:
            return bbox1, bbox2
    return None, None





