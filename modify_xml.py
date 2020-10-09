#!/usr/bin/env python
# encoding:utf-8
"""
author: liusili
@contact: liusili@unionbigdata.com
@software:
@file: modify_xml
@time: 2020/7/15
@desc: 
"""
import os
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


def replace_category(xml_path, old_cat, new_cat):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for obj in root.findall('object'):
        name = get_and_check(obj, 'name', 1).text
        if name == old_cat:
            get_and_check(obj, 'name', 1).text = new_cat
    tree.write(xml_path)
    print('finish.')


def check_category(xml_path, category):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for obj in root.findall('object'):
        name = get_and_check(obj, 'name', 1).text
        if name == category:
            return 1
    return 0


def find_wrong_tag(sample_root, category):
    for file in os.listdir(sample_root):
        if os.path.splitext(file)[-1] == '.xml':
            xml_path = os.path.join(sample_root, file)
            if not check_category(xml_path, category):
                print(file)
    print('finish.')


if __name__ == '__main__':
    xml_path = r'D:\Working\Tianma\Mask-FMM\data\0800\ADD_0819\0821\PO02\FMM_0821_071.xml'
    old_cat = 'normal'
    new_cat = 'E03_O'
    # old_cat = 'E03_O'
    # new_cat = 'normal'
    replace_category(xml_path, old_cat, new_cat)

    # sample_root = r'D:\Working\Tianma\Mask-FMM\data\mask_0713\PO03'
    # find_wrong_tag(sample_root, 'PO03')