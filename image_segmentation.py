#!/usr/bin/env python
# encoding:utf-8
"""
author: liusili
@contact: liusili@unionbigdata.com
@software:
@file: image_segmentation
@time: 9/30/2020
@desc: 
"""
import os
import cv2
import numpy as np
from collections import namedtuple
import xml.etree.ElementTree as ET
from xml.dom.minidom import Document


def generate_xml(image_info, tag_list, out_dir=None):
    """
    :param out_dir:
    :info: 输出xml文件
    :param image_info:
    :param tag_list:
    :return:
    """
    img_name = image_info['file_name']
    img_h = str(image_info['height'])
    img_w = str(image_info['width'])
    img_path = image_info['path']

    folder = img_path.split('\\')[-2]

    doc = Document()
    orderpack = doc.createElement("annotation")
    doc.appendChild(orderpack)

    objectfolder = doc.createElement("folder")
    objectcontenttext = doc.createTextNode(folder)
    objectfolder.appendChild(objectcontenttext)
    orderpack.appendChild(objectfolder)

    objectfilename = doc.createElement("filename")
    objectcontenttext = doc.createTextNode(img_name)
    objectfilename.appendChild(objectcontenttext)
    orderpack.appendChild(objectfilename)

    objectpath = doc.createElement("path")
    objectcontenttext = doc.createTextNode(img_path)
    objectpath.appendChild(objectcontenttext)
    orderpack.appendChild(objectpath)

    objectsource = doc.createElement("source")
    orderpack.appendChild(objectsource)

    objectdatabase = doc.createElement("database")
    objectdatabasetext = doc.createTextNode('Unknown')
    objectdatabase.appendChild(objectdatabasetext)
    objectsource.appendChild(objectdatabase)

    objectsize = doc.createElement("size")
    orderpack.appendChild(objectsize)

    objectwidth = doc.createElement("width")
    objectwidthtext = doc.createTextNode(img_w)
    objectwidth.appendChild(objectwidthtext)
    objectsize.appendChild(objectwidth)

    objectheight = doc.createElement("height")
    objectheighttext = doc.createTextNode(img_h)
    objectheight.appendChild(objectheighttext)
    objectsize.appendChild(objectheight)

    objectdepth = doc.createElement("depth")
    objectdepthtext = doc.createTextNode('3')
    objectdepth.appendChild(objectdepthtext)
    objectsize.appendChild(objectdepth)

    objectcontent = doc.createElement("segmented")
    objectcontenttext = doc.createTextNode('0')
    objectcontent.appendChild(objectcontenttext)
    orderpack.appendChild(objectcontent)

    for tag in tag_list:
        bbox = tag.bbox
        category = tag.name
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[2]
        ymax = bbox[3]

        xmin = str(int(xmin))
        ymin = str(int(ymin))
        xmax = str(int(xmax))
        ymax = str(int(ymax))

        objectobject = doc.createElement("object")
        orderpack.appendChild(objectobject)

        objectname = doc.createElement("name")
        objectcontenttext = doc.createTextNode(str(category))
        objectname.appendChild(objectcontenttext)
        objectobject.appendChild(objectname)

        objectpose = doc.createElement("pose")
        objectcontenttext = doc.createTextNode('Unspecified')
        objectpose.appendChild(objectcontenttext)
        objectobject.appendChild(objectpose)

        objecttruncated = doc.createElement("truncated")
        objectcontenttext = doc.createTextNode('0')
        objecttruncated.appendChild(objectcontenttext)
        objectobject.appendChild(objecttruncated)

        objectdifficult = doc.createElement("difficult")
        objectcontenttext = doc.createTextNode('0')
        objectdifficult.appendChild(objectcontenttext)
        objectobject.appendChild(objectdifficult)

        objectbndbox = doc.createElement("bndbox")
        objectobject.appendChild(objectbndbox)

        objectxmin = doc.createElement("xmin")
        objectcontenttext = doc.createTextNode(xmin)
        objectxmin.appendChild(objectcontenttext)
        objectbndbox.appendChild(objectxmin)

        objectymin = doc.createElement("ymin")
        objectcontenttext = doc.createTextNode(ymin)
        objectymin.appendChild(objectcontenttext)
        objectbndbox.appendChild(objectymin)

        objectxmax = doc.createElement("xmax")
        objectcontenttext = doc.createTextNode(xmax)
        objectxmax.appendChild(objectcontenttext)
        objectbndbox.appendChild(objectxmax)

        objectymax = doc.createElement("ymax")
        objectcontenttext = doc.createTextNode(ymax)
        objectymax.appendChild(objectcontenttext)
        objectbndbox.appendChild(objectymax)

    if not out_dir:
        xml_file = os.path.splitext(img_path)[0] + '.xml'
    else:
        xml_file = os.path.join(out_dir, os.path.splitext(img_name)[0] + '.xml')
    f = open(xml_file, 'w')
    doc.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
    f.close()


def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = np.array(img, dtype='uint8')
    img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)[1]
    return img


def get_contours(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_lst = []
    bbox_lst = []
    for c in contours:
        bbox = cv2.boundingRect(c)
        area = bbox[2] * bbox[3]
        if area < 1000:
            continue
        area_lst.append(area)
        bbox_lst.append(bbox)

    area_lst = np.array(area_lst)
    bbox_lst = np.array(bbox_lst)
    if len(area_lst) > 0:
        # 取框面积的中位数的 +-20% 为标准
        out_lst = bbox_lst[(np.median(area_lst) * 1.2 > area_lst)
                           & (area_lst > np.median(area_lst) * 0.8)]
    else:
        out_lst = []
    return out_lst


def get_standard_scale(section_lst, padding=50):
    s_width = int(np.median(section_lst[:, 2]))
    s_height = int(np.median(section_lst[:, 3]))
    s_width = round((s_width + padding)/padding) * padding
    s_height = round((s_height + padding)/padding) * padding
    return s_width, s_height


def get_standard_section(bbox_lst, s_width, s_height):
    standard_lst = []
    for bbox in bbox_lst:
        xmin, ymin, w, h = bbox
        padding = (s_width - w) // 2
        xmin = xmin - padding
        xmax = xmin + s_width

        padding = (s_height - h) // 2
        ymin = ymin - padding
        ymax = ymin + s_height

        standard_lst.append((xmin, ymin, xmax, ymax))
    return standard_lst


def get_first_section(section_lst, s_width):
    base = min(list(map(lambda x: x[0], section_lst)))
    tmp = list(filter(lambda x: x[0] < base+0.5*s_width, section_lst))
    return min(tmp, key=lambda x: x[1])


def find_period(section_lst, s_height, s_width):
    x_lst = [x[0] for x in section_lst]
    y_lst = [x[1] for x in section_lst]
    x_lst = sorted(x_lst)
    x_lst = np.array(x_lst)
    y_lst = sorted(y_lst)
    y_lst = np.array(y_lst)

    # 横向周期分析
    diff_x = x_lst[1:] - x_lst[:-1]
    period_x_lst = diff_x[(diff_x/s_width > 0.8) & (diff_x/s_width < 1.2)]
    period_x = max(period_x_lst)
    # 纵向周期分析
    diff_y = y_lst[1:] - y_lst[:-1]
    period_y_lst = diff_y[(diff_y/s_height > 0.8) & (diff_y/s_height < 1.2)]
    period_y = max(period_y_lst)
    return period_x, period_y


def in_section(bbox, bbox_lst):
    c_x = (bbox[2] + bbox[0])/2
    c_y = (bbox[3] + bbox[1])/2
    for b in bbox_lst:
        xmin, ymin, xmax, ymax = b
        if xmin < c_x < xmax and ymin < c_y < ymax:
            return True, b
    return False, bbox


# 在检测到的点灯区小于95时启动
def recover_section(first_section, section_lst, width, height,
                    s_height, s_width, period_x, period_y):
    xmin, ymin, _, _, = first_section
    start_x = xmin % period_x
    start_y = ymin % period_y
    x = start_x
    y = start_y
    recover_lst = []
    while y + s_height < height:
        while x + s_width < width:
            bbox = (x, y, x+s_width, y+s_height)
            section_check = in_section(bbox, section_lst)
            if section_check[0]:
                x, y, _, _ = section_check[1]
            else:
                recover_lst.append(bbox)
            x += period_x
        x = start_x
        y += period_y
    return recover_lst


def get_tags_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    Tag = namedtuple('Tag', ['bbox', 'name'])
    out_lst = []
    for obj in root.findall('object'):
        cat = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        tag = Tag([xmin, ymin, xmax, ymax], cat)
        out_lst.append(tag)
    return out_lst


def get_intersection(bbox1, bbox2):
    xmin = max(bbox1[0], bbox2[0])
    ymin = max(bbox1[1], bbox2[1])
    xmax = min(bbox1[2], bbox2[2])
    ymax = min(bbox1[3], bbox2[3])
    if xmin >= xmax or ymin >= ymax:
        return None
    else:
        return [xmin, ymin, xmax, ymax]


def update_bbox(section_lst, tags_lst):
    out_lst = []
    Tag = namedtuple('Tag', ['bbox', 'name'])
    for section in section_lst:
        new_tag_lst = []
        x = section[0]
        y = section[1]
        for tag in tags_lst:
            bbox = tag.bbox
            inter = get_intersection(section, bbox)
            if inter:
                cat = tag.name
                new_bbox = [inter[0]-x, inter[1]-y, inter[2]-x, inter[3]-y]
                new_tag = Tag(new_bbox, cat)
                new_tag_lst.append(new_tag)
        out_lst.append(new_tag_lst)
    return out_lst


def image_segmentation(img):
    """
    对点灯机图片进行切分，以及对于缺失部分区域进行恢复
    :param img:
    :return: 所有分割区域的坐标信息以及恢复区域的坐标信息 List[xmin, ymin, xmax, ymax]
    """
    height, width = img.shape
    out_lst = get_contours(img)
    s_width, s_height = get_standard_scale(out_lst)
    section_lst = get_standard_section(out_lst, s_width, s_height)
    period_x, period_y = find_period(section_lst, s_height, s_width)
    first_section = get_first_section(section_lst, s_width)
    recover_lst = recover_section(first_section, section_lst, width, height,
                                  s_height, s_width, period_x, period_y)
    return section_lst, recover_lst


def data_segmentation(img_path, xml_path, file_name, out_path):
    img_raw = cv2.imread(img_path)
    img = preprocess(img_raw)
    section_lst, recover_lst = image_segmentation(img)
    tags_lst = get_tags_from_xml(xml_path)
    section_tags = update_bbox(section_lst, tags_lst)
    recover_tags = update_bbox(recover_lst, tags_lst)

    # 保存路径
    normal_path = os.path.join(out_path, 'normal')
    recover_path = os.path.join(out_path, 'abnormal')
    tag_path = os.path.join(out_path, 'tag')
    os.makedirs(normal_path, exist_ok=True)
    os.makedirs(recover_path, exist_ok=True)
    os.makedirs(tag_path, exist_ok=True)

    ind = 1
    for i, tag_lst in enumerate(section_tags):
        img_name = file_name + '_{}.jpg'.format(ind)
        section = section_lst[i]
        sec = img_raw[section[1]:section[3], section[0]:section[2], :]
        ind += 1
        if tag_lst:
            file_path = os.path.join(tag_path, img_name)
            h, w, _ = sec.shape
            image_info = {'file_name': img_name, 'path': file_path, 'height': h, 'width': w}
            generate_xml(image_info, tag_lst)
        else:
            file_path = os.path.join(normal_path, img_name)
        cv2.imwrite(file_path, sec)

    for i, tag_lst in enumerate(recover_tags):
        img_name = file_name + '_{}.jpg'.format(ind)
        section = recover_lst[i]
        sec = img_raw[section[1]:section[3], section[0]:section[2], :]

        ind += 1
        if tag_lst:
            file_path = os.path.join(tag_path, img_name)
            h, w, _ = sec.shape
            image_info = {'file_name': img_name, 'path': file_path, 'height': h, 'width': w}
            generate_xml(image_info, tag_lst)
        else:
            file_path = os.path.join(recover_path, img_name)
        cv2.imwrite(file_path, sec)



if __name__ == '__main__':
    out_path = r'C:\Users\OPzealot\Desktop\LIGHTER'
    img_path = r'C:\Users\OPzealot\Desktop\LIGHTER\1.jpg'
    xml_path = r'C:\Users\OPzealot\Desktop\LIGHTER\1.xml'
    file_name = '1'
    data_segmentation(img_path, xml_path, file_name, out_path)
    print('finish')
