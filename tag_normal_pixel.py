#!/usr/bin/env python
# encoding:utf-8
"""
author: liusili
@contact: liusili@unionbigdata.com
@software:
@file: tag_normal_pixel.py
@time: 2020/6/4
@desc: 
"""

import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
from glob import glob
from tqdm import tqdm
from generate_xml import generate_xml


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


def pretty_xml(element, indent, newline, level=0):  # element为传进来的Element类，参数indent用于缩进，newline用于换行
    if element:  # 判断element是否有子元素
        if element.text is None or element.text.isspace():  # 如果element的text没有内容
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
    # else:  # 此处两行如果把注释去掉，Element的text也会另起一行
    # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
    temp = list(element)  # 将element转成list
    for subelement in temp:
        if temp.index(subelement) < (len(temp) - 1):  # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
            subelement.tail = newline + indent * (level + 1)
        else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
            subelement.tail = newline + indent * level
        pretty_xml(subelement, indent, newline, level=level + 1)  # 对子元素进行递归操作


def add_bbox_info(tree, category, bbox_lst):
    root = tree.getroot()
    for bbox in bbox_lst:
        x,y,w,h = bbox
        xmin = x
        xmax = x+w
        ymin = y
        ymax = y+h

        SubElement_object = ET.SubElement(root, 'object')
        SubElement_name = ET.SubElement(SubElement_object, 'name')
        SubElement_name.text = category

        SubElement_pose = ET.SubElement(SubElement_object, 'pose')
        SubElement_pose.text = 'Unspecified'

        SubElement_truncated = ET.SubElement(SubElement_object, 'truncated')
        SubElement_truncated.text = '0'

        SubElement_difficult = ET.SubElement(SubElement_object, 'difficult')
        SubElement_difficult.text = '0'

        SubElement_bndbox = ET.SubElement(SubElement_object, 'bndbox')
        SubElement_xmin = ET.SubElement(SubElement_bndbox, 'xmin')
        SubElement_xmin.text = str(xmin)
        SubElement_xmin = ET.SubElement(SubElement_bndbox, 'ymin')
        SubElement_xmin.text = str(ymin)
        SubElement_xmin = ET.SubElement(SubElement_bndbox, 'xmax')
        SubElement_xmin.text = str(xmax)
        SubElement_xmin = ET.SubElement(SubElement_bndbox, 'ymax')
        SubElement_xmin.text = str(ymax)
    pretty_xml(root, '\t', '\n')
    return tree


def tag_normal_bbox(img_path, category='normal'):
    addition = False
    img_raw = cv2.imread(img_path)
    height = img_raw.shape[0]
    width = img_raw.shape[1]
    file_name = os.path.splitext(img_path)[0]
    xml_path = file_name + '.xml'

    img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
    img = cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)

    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = np.array(img, dtype='uint8')
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    if os.path.isfile(xml_path):
        addition = True
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            bbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(get_and_check(bbox, 'xmin', 1).text)
            ymin = int(get_and_check(bbox, 'ymin', 1).text)
            xmax = int(get_and_check(bbox, 'xmax', 1).text)
            ymax = int(get_and_check(bbox, 'ymax', 1).text)

            img[ymin:ymax, xmin:xmax] = 0

    kernel = np.ones((3, 3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    area_lst = []
    bbox_lst = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < 30:
            continue
        bbox = cv2.boundingRect(c)
        x,y,w,h = bbox
        new_x = max(0, x-5)
        new_w = min(w+10, width - new_x)
        new_y = max(0, y-5)
        new_h = min(h+10, height - new_y)
        bbox = [new_x, new_y, new_w, new_h]

        area_lst.append(area)
        bbox_lst.append(bbox)

    area_lst = np.array(area_lst)
    bbox_lst = np.array(bbox_lst)

    if len(area_lst) > 0:
        out_lst = bbox_lst[area_lst > max(area_lst) * 0.1]
    else:
        out_lst = []
    if addition:
        tree = add_bbox_info(tree, category, out_lst)
        tree.write(xml_path)

    else:
        image_info = {}
        image_info['file_name'] = img_path.split('\\')[-1]
        image_info['path'] = img_path
        image_info['height'] = height
        image_info['width'] = width
        generate_xml(image_info, category, out_lst)


def main(root_path, category):
    img_lst = glob(root_path + '/*/*.jpg')
    pbar = tqdm(img_lst)
    for img_path in pbar:
        tqdm.write('[正在打标] {}'.format(img_path))
        tag_normal_bbox(img_path, category)
        pbar.set_description('Processing')
    print('FINISH.')


if __name__ == '__main__':
    root_path = r'D:\Working\Tianma\Mask-FMM\data\0800\ADD_0819\0821'
    main(root_path, 'normal')
