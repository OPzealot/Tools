#!/usr/bin/env python
# encoding:utf-8
"""
author: liusili
@contact: liusili@unionbigdata.com
@software:
@file: export_category
@time: 8/26/2020
@desc: 
"""
import json
import os


def export_category_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        infoDict = json.load(f)
    catLst = []
    for info in infoDict['categories']:
        category = info['name'] + '\n'
        catLst.append(category)

    file_name = json_path.split('\\')[-1]
    dir_path = json_path.split(file_name)[0]
    out_path = os.path.join(dir_path, 'classes.txt')
    with open(out_path, 'w') as f:
        f.writelines(catLst)


if __name__ == '__main__':
    src_path = r"D:\Working\Tianma\54902\work_dir\20201218_all\train.json"
    export_category_from_json(src_path)