#!/usr/bin/env python
# encoding:utf-8
"""
author: liusili
@contact: liusili@unionbigdata.com
@software:
@file: recover_pixel
@time: 2020/6/22
@desc: 
"""
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm


def get_standard(pixel_lst, height, width):
    width_lst = []
    height_lst = []
    standard_lst = []
    for bbox in pixel_lst:
        if bbox[0] > 2 and bbox[2] < width - 2:
            width_lst.append(bbox[2] - bbox[0])
        if bbox[1] > 2 and bbox[3] < height -2:
            height_lst.append(bbox[3] - bbox[1])

    s_width = np.median(width_lst)
    s_height = np.median(height_lst)

    if s_width and s_height:
        for bbox in pixel_lst:
            b_width = bbox[2] - bbox[0]
            b_height = bbox[3] - bbox[1]
            if b_width > 0.9 * s_width and b_height > 0.9 * s_height:
                standard_lst.append(bbox)
    return s_height, s_width, standard_lst


def get_standard_bbox(bbox_lst, height, width, s_height, s_width):
    s_bbox_lst = []
    for bbox in bbox_lst:
        xmin, ymin, xmax, ymax = bbox
        b_width = xmax - xmin
        b_height = ymax - ymin
        if b_width < s_width:
            if xmin < 5:
                xmin = xmax - s_width
            elif xmax > width - 5:
                xmax = xmin + s_width
            else:
                padding = (s_width - b_width) / 2
                xmin = xmin - padding
                xmax = xmax + padding

        if b_height < s_height:
            if ymin < 5:
                ymin = ymax - s_height
            elif ymax > height - 5:
                ymax = ymin + s_height
            else:
                padding = (s_height - b_height) / 2
                ymin = ymin - padding
                ymax = ymax + padding

        s_bbox_lst.append((xmin, ymin, xmax, ymax))
    return s_bbox_lst


def find_period(s_pixel_lst, height, width, s_height, s_width):
    x_lst = [x[0] for x in s_pixel_lst]
    y_lst = [x[1] for x in s_pixel_lst]
    x_lst = sorted(x_lst)
    x_lst = np.array(x_lst)
    y_lst = sorted(y_lst)
    y_lst = np.array(y_lst)

    # 横向周期分析
    diff_x = x_lst[1:] - x_lst[:-1]
    period_x_lst = diff_x[(diff_x/s_width > 0.8) & (diff_x/s_width < 3.5)]
    period_x = max(period_x_lst)
    if period_x > 2 * s_width:
        period_x = period_x / 2

    # 纵向周期分析
    diff_y = y_lst[1:] - y_lst[:-1]
    period_y_lst = diff_y[(diff_y/s_height > 0.5) & (diff_y/s_height < 2)]
    period_y = max(period_y_lst)
    if period_y > 1.5 * s_height:
        period_y = period_y / 2

    return period_x, period_y


def get_first_pixel(s_pixel_lst, s_width):
    base = min(list(map(lambda x: x[0], s_pixel_lst)))
    tmp = list(filter(lambda x: x[0] < base+0.5*s_width, s_pixel_lst))
    return min(tmp,key=lambda x: x[1])


def backward_pixel(start_bbox, height, period_x, period_y):
    backward_lst = []
    xmin, ymin, xmax, ymax = start_bbox
    weight = -1
    while xmax > 0 and ymax > 0 and ymin < height:
        # 判断是否在顶部
        if ymax - 2*period_y < 0 and weight == -1:
            xmin = xmin - period_x
            xmax = xmax - period_x
            if ymax - period_y > 0:
                ymin = ymin - period_y
                ymax = ymax - period_y
            else:
                ymin = ymin + period_y
                ymax = ymax + period_y
            # 翻转
            weight = weight * -1
            if xmax > 0:
                backward_lst.append((xmin, ymin, xmax, ymax))
            continue

        # 正常向上或向下平移
        ymin = ymin + weight * period_y * 2
        ymax = ymax + weight * period_y * 2

        # 判断是否到达底部
        if ymin + 2*period_y > height:
            if ymin < height:
                backward_lst.append((xmin, ymin, xmax, ymax))

            xmin = xmin - period_x
            xmax = xmax - period_x
            if ymin + period_y < height:
                ymin = ymin + period_y
                ymax = ymax + period_y
            else:
                ymin = ymin - period_y
                ymax = ymax - period_y
            # 翻转
            weight = weight * -1
            if xmax > 0:
                backward_lst.append((xmin, ymin, xmax, ymax))
            continue

        backward_lst.append((xmin, ymin, xmax, ymax))
    return backward_lst


def in_pixel(bbox, bbox_lst):
    c_x = (bbox[2] + bbox[0])/2
    c_y = (bbox[3] + bbox[1])/2
    for b in bbox_lst:
        xmin, ymin, xmax, ymax = b
        if xmin < c_x < xmax and ymin < c_y < ymax:
            return True, b
    return False, bbox


def forward_pixel(start_bbox, pixel_lst, height, width, period_x, period_y):
    forward_lst = []
    xmin, ymin, xmax, ymax = start_bbox
    weight = 1
    while xmin < width and ymax > 0 and ymin < height:
        # 判断是否在底部
        if ymin + 2*period_y > height and weight == 1:
            xmin = xmin + period_x
            xmax = xmax + period_x
            if ymin + period_y < height:
                ymin = ymin + period_y
                ymax = ymax + period_y
            else:
                ymin = ymin - period_y
                ymax = ymax - period_y
            # 翻转
            weight = weight * -1
            if xmin < width:
                bbox = (xmin, ymin, xmax, ymax)
                pixel_check = in_pixel(bbox, pixel_lst)
                if pixel_check[0]:
                    xmin, ymin, xmax, ymax = pixel_check[1]
                else:
                    forward_lst.append(bbox)
            continue
        # 正常向上或向下平移
        ymin = ymin + weight * period_y * 2
        ymax = ymax + weight * period_y * 2

        # 判断是否到达顶部
        if ymax - 2*period_y < 0:
            if ymax > 0:
                if xmin < width:
                    bbox = (xmin, ymin, xmax, ymax)
                    pixel_check = in_pixel(bbox, pixel_lst)
                    if pixel_check[0]:
                        xmin, ymin, xmax, ymax = pixel_check[1]
                    else:
                        forward_lst.append(bbox)
            xmin = xmin + period_x
            xmax = xmax + period_x
            if ymax - period_y > 0:
                ymin = ymin - period_y
                ymax = ymax - period_y
            else:
                ymin = ymin + period_y
                ymax = ymax + period_y
            # 翻转
            weight = weight * -1
            if xmin < width:
                bbox = (xmin, ymin, xmax, ymax)
                pixel_check = in_pixel(bbox, pixel_lst)
                if pixel_check[0]:
                    xmin, ymin, xmax, ymax = pixel_check[1]
                else:
                    forward_lst.append(bbox)
            continue

        bbox = (xmin, ymin, xmax, ymax)
        pixel_check = in_pixel(bbox, pixel_lst)
        if pixel_check[0]:
            xmin, ymin, xmax, ymax = pixel_check[1]
        else:
            forward_lst.append(bbox)
    return forward_lst


def recover_pixel(img, pixel_lst):
    height = img.shape[0]
    width = img.shape[1]
    s_height, s_width, standard_lst = get_standard(pixel_lst, height, width)
    s_pixel_lst = get_standard_bbox(pixel_lst, height, width, s_height, s_width)
    period_x, period_y = find_period(s_pixel_lst, height, width, s_height, s_width)
    first_pixel = get_first_pixel(s_pixel_lst, s_width)

    backward_lst = backward_pixel(first_pixel, height, period_x, period_y)
    forward_lst = forward_pixel(first_pixel, s_pixel_lst, height, width, period_x, period_y)
    recover_lst = backward_lst + forward_lst
    return recover_lst, standard_lst, s_height, s_width


def draw_bbox(img, bbox_lst, color=(0, 255, 0)):
    for bbox in bbox_lst:
        xmin = int(bbox[0])
        ymin = int(bbox[1])
        xmax = int(bbox[2])
        ymax = int(bbox[3])
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 1)
    return img


def get_pixel_area(gray, bbox, padding=5):
    height, width = gray.shape
    xmin, ymin, xmax, ymax = bbox
    xmin = max(xmin, 0)
    ymin = max(ymin, 0)
    xmax = min(xmax, width)
    ymax = min(ymax, height)
    bbox_area = (xmax - xmin) * (ymax - ymin)

    xmin = int(max(xmin - padding, 0))
    ymin = int(max(ymin - padding, 0))
    xmax = int(min(xmax + padding, width))
    ymax = int(min(ymax + padding, height))

    pixel = gray[ymin:ymax, xmin:xmax]
    area = len(pixel[pixel == 255])
    return area, bbox_area


def get_normal_area(gray, standard_lst):
    area_lst = []
    for bbox in standard_lst:
        area, _ = get_pixel_area(gray, bbox, 0)
        area_lst.append(area)
    s_area = min(area_lst)
    return s_area


def abnormal_pixel(gray, recover_lst, standard_lst, s_height, s_width, area_thr=0):
    normal_area = get_normal_area(gray, standard_lst)

    percentage_lst = []
    for bbox in recover_lst:
        area, bbox_area = get_pixel_area(gray, bbox, 5)
        percentage = area / normal_area
        ratio = min(bbox_area / (s_height * s_width),1)
        percentage = percentage / ratio
        if area_thr < ratio < 0.8:
            if percentage < 0.9:
                percentage_lst.append((1-percentage)*ratio)
            else:
                percentage_lst.append(0)
            continue
        elif ratio <= area_thr:
            percentage_lst.append(0)
            continue
        percentage_lst.append(1-percentage)
    return percentage_lst


def count_abnormal(percentage_lst):
    cnt = 0
    for p in percentage_lst:
        if p != 1:
            cnt += 1
    return cnt


def text_image(img, cnt, percentage_lst, recover_lst):
    height = img.shape[0]
    width = img.shape[1]
    font_size = round(height / 250, 1)
    font_length = int(font_size * 14)
    font_height = int(font_size * 20)
    thickness = int(font_size * 0.6 + 1)

    cnt_text_cord = (1, font_height)
    cv2.putText(img, 'Count: {}'.format(cnt), cnt_text_cord,
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                font_size, (0, 0, 255), thickness)

    for i, bbox in enumerate(recover_lst):
        percentage = percentage_lst[i]
        if percentage == 1:
            continue
        xmin = max(int(bbox[0]), 0)
        ymin = max(int(bbox[1]), 0)

        p_cord = (min(xmin, width - font_length * 3), ymin + font_height)

        cv2.putText(img, '{:.0f}%'.format(percentage*100), p_cord,
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    font_size, (0, 0, 255), thickness)
    return img


def main(sample_root, results_path, pixel_path, out_path):
    with open(pixel_path, 'rb') as f:
        pixel_results = pickle.load(f)
    results_df = pd.read_excel(results_path, index_col=0)
    pbar = tqdm(results_df.index)
    for i in pbar:
        img_name = results_df.loc[i, 'image_name']
        category = results_df.loc[i, 'category']
        if category == 'FALSE' or category == 'PO01':
            continue
        img_path = os.path.join(sample_root, category, img_name)
        pixel_lst = pixel_results[i-1]

        img = cv2.imread(img_path)
        try:
            recover_lst, standard_lst, s_height, s_width = recover_pixel(img, pixel_lst)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            gray = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)[1]
            percentage_lst = abnormal_pixel(gray, recover_lst, standard_lst, s_height, s_width)
            cnt = count_abnormal(percentage_lst)
        except Exception as e:
            tqdm.write('\n [Error]:{} \n [category] {}; [image] {}'.format(e, category, img_name))
            recover_lst = []
            percentage_lst = []
            cnt = 0
        img = draw_bbox(img, pixel_lst)
        img = draw_bbox(img, recover_lst, color=(0, 0, 255))
        img = text_image(img, cnt, percentage_lst, recover_lst)
        cat_path = os.path.join(out_path, category)
        os.makedirs(cat_path, exist_ok=True)
        cv2.imwrite(os.path.join(cat_path, img_name), img)


def recover(img_name, results_path, pixel_path, out_path):
    with open(pixel_path, 'rb') as f:
        pixel_results = pickle.load(f)
    results_df = pd.read_excel(results_path, index_col=0)
    i = results_df[results_df['image_name'] == img_name].index[0]
    category = results_df.loc[i, 'category']
    img_path = os.path.join(sample_root, category, img_name)
    pixel_lst = pixel_results[i-1]
    img = cv2.imread(img_path)
    # recover_lst, standard_lst, s_height, s_width = recover_pixel(img, pixel_lst)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    gray = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)[1]
    # percentage_lst = abnormal_pixel(gray, recover_lst, standard_lst, s_height, s_width)
    # cnt = count_abnormal(percentage_lst)

    img = draw_bbox(img, pixel_lst)
    # img = draw_bbox(img, recover_lst, color=(0, 0, 255))
    # img = text_image(img, cnt, percentage_lst, recover_lst)
    cv2.imwrite(os.path.join(out_path, img_name), img)


if __name__ == '__main__':
    pixel_path = r'D:\Working\Tianma\Mask-FMM\TEST\result\out\pixel_results.pkl'
    results_path = r'D:\Working\Tianma\Mask-FMM\TEST\result\out\deploy_results.xlsx'
    sample_root = r'D:\Working\Tianma\Mask-FMM\TEST\data\testset_0707'
    out_path = r'C:\Users\OPzealot\Desktop\fmm_test_3'
    # main(sample_root, results_path, pixel_path, out_path)

    img_name = r'fmm__0705.jpg'
    recover(img_name, results_path, pixel_path, out_path)
