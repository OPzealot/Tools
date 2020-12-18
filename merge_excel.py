#!/usr/bin/env python
# encoding:utf-8
"""
author: liusili
@l@icense: (C) Copyright 2020, Union Big Data Co. Ltd. All rights reserved.
@contact: liusili@unionbigdata.com
@software:
@file: merge_excel
@time: 2020/3/27
@desc: 
"""
import pandas as pd
import numpy as np
import os


def combine_table(table_1, table_2, table_3, out_path):
    df_1 = pd.read_excel(table_1, sheet_name='code')
    df_2 = pd.read_excel(table_2, sheet_name='code')
    df_3 = pd.read_excel(table_3, sheet_name='code')

    columns = ['glass_id', 'panel_id', 'category', 'score']
    out_df = pd.DataFrame(columns=columns)
    id = 0

    for row in df_1.itertuples():
        image_name = getattr(row, 'image_name')
        glass = image_name.split('.')[0]

        panel = getattr(row, 'panel_id')
        infer = getattr(row, 'category')
        score = getattr(row, 'score')

        id += 1
        out_df.loc[id] = [glass, panel, infer, score]

    for row in df_2.itertuples():
        image_name = getattr(row, 'image_name')
        glass = image_name.split('.')[0]

        panel = getattr(row, 'panel_id')
        infer = getattr(row, 'category')
        score = getattr(row, 'score')

        id += 1
        out_df.loc[id] = [glass, panel, infer, score]

    for row in df_3.itertuples():
        image_name = getattr(row, 'image_name')
        glass = image_name.split('_')[0]

        panel = getattr(row, 'panel_id')
        infer = getattr(row, 'category')
        score = getattr(row, 'score')

        id += 1
        out_df.loc[id] = [glass, panel, infer, score]

    out_df.to_excel(out_path, sheet_name='results')
    print('[FINISH] Table Combination.')



def merge_excel(category, root_path, out_path):
    """
    :info: 默认是将‘0’类判断为预测正确
    :param category:
    :param root_path:
    :param out_path:
    :return:
    """
    n = len(category)
    category = sorted(category)
    df = pd.DataFrame(np.zeros([n, n], dtype='uint'), index=category, columns=category)

    for table in os.listdir(root_path):
        table_path = os.path.join(root_path, table)
        df_tmp = pd.read_excel(table_path, index_col=0, sheet_name='图表')
        for col in df_tmp.columns:
            for row in df_tmp.columns:
                df[col][row] = int(df[col][row]) + int(df_tmp[col][row])

    predict_sum = []
    ori_sum = []
    precision_lst = []
    recall_lst = []
    for i in category:
        predict_cnt = sum(df[i])
        ori_cnt = sum(df.loc[i]) + df[i]['0']
        predict_sum.append(predict_cnt)
        ori_sum.append(ori_cnt)
        correct = df[i][i] + df[i]['0']
        precision = round(correct / predict_cnt, 3)
        recall = round(correct / ori_cnt, 3)
        precision_lst.append(precision)
        recall_lst.append(recall)

    df.loc['预测合计'] = predict_sum
    df.loc['准确率'] = precision_lst
    ori_sum = ori_sum + [None] * 2
    recall_lst = recall_lst + [None] * 2
    df['判图合计'] = ori_sum
    df['召回率'] = recall_lst

    df.to_excel(out_path, sheet_name='统计')
    print('[FINISH] Saving file to {}'.format(out_path))


if __name__ == '__main__':
    table_1 = r"E:\Working\Visionox\V2_lighter\file\POC_report\DemoReport\test_1126\2CEE01_results.xlsx"
    table_2 = r"E:\Working\Visionox\V2_lighter\file\POC_report\DemoReport\test_1126\2CIL01_results.xlsx"
    table_3 = r"E:\Working\Visionox\V2_lighter\file\POC_report\DemoReport\test_1126\Mapping_results.xlsx"
    out_path = r"E:\Working\Visionox\V2_lighter\file\POC_report\DemoReport\test_1126\results.xlsx"
    combine_table(table_1, table_2, table_3, out_path)