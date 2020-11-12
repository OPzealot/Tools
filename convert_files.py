#!/usr/bin/env python
# encoding:utf-8
"""
author: liusili
@contact: liusili@unionbigdata.com
@software:
@file: convert_files
@time: 2020/8/4
@desc: 
"""


def file2list(file_path):
    cat_lst = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            category = line.strip('\n')
            cat_lst.append(category)
    print(cat_lst)


if __name__ == '__main__':
    file_path = r'D:\Working\Tianma\54902\deploy\classes.txt'
    file2list(file_path)