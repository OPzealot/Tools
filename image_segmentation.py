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


def preprocess(img_path):
    img = cv2.imread(img_path, 0)
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
    return min(tmp,key=lambda x: x[1])


def find_period(section_lst,s_height, s_width):
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
