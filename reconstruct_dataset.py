#!/usr/bin/env python
# encoding:utf-8
"""
author: liusili
@contact: liusili@unionbigdata.com
@software:
@file: reconstruct_dataset
@time: 2020/4/27
@desc: 
"""
import os
import shutil
from glob import glob
from tqdm import tqdm


def reconstruct_dataset(root_path, img_format='.jpg'):
    for root, _, file_lst in os.walk(sample_root):
        if len(file_lst) > 0:
            pbar = tqdm(file_lst)
            for file in pbar:
                if os.path.splitext(file)[-1] == img_format:
                    info_lst = root.split('\\')
                    cat = info_lst[-1]
                    category = cat.split('-')[-1]

                    file_path = os.path.join(root, file)
                    new_code_path = os.path.join(root_path + '_new', category)
                    os.makedirs(new_code_path, exist_ok=True)
                    new_file_path = os.path.join(new_code_path, file)
                    shutil.copy(file_path, new_file_path)
    print('FINISH')


def reconstruct_by_product(root_path):
    for code_name in tqdm(os.listdir(root_path)):
        code = code_name.rsplit('-', 1)[-1]
        code_path = os.path.join(root_path, code_name)

        for product in os.listdir(code_path):
            new_code_path = os.path.join(root_path + '_product_{}'.format(product), code)
            os.makedirs(new_code_path, exist_ok=True)
            product_path = os.path.join(code_path, product)
            for file in os.listdir(product_path):
                file_path = os.path.join(product_path, file)
                new_file_path = os.path.join(new_code_path, file)
                shutil.copy(file_path, new_file_path)

    print('FINISH')


def reconstruct_testset(root_path, out_path):
    img_lst = glob(root_path + '/*/*/*/*.jpg')
    os.makedirs(out_path, exist_ok=False)
    for img_path in img_lst:
        img_name = img_path.split('\\')[-1]
        new_path = os.path.join(out_path, img_name)
        shutil.copy(img_path, new_path)

    print('FINISH')


def extract_wrong_data(root_path, out_path, category):
    img_lst = glob(root_path + '/*/{}/*.jpg'.format(category))
    os.makedirs(out_path, exist_ok=False)
    for img_path in img_lst:
        img_name = img_path.split('\\')[-1]
        origin = img_path.split('\\')[-3]
        if origin != category:
            new_path = os.path.join(out_path, img_name)
            shutil.copy(img_path, new_path)
    print('FINISH.')


if __name__ == '__main__':
    # root_path = r'D:\Working\Tianma\13902\TEST\0608\test_3\out'
    # out_path = r'D:\Working\Tianma\13902\TEST\0608\test_3\out_1'
    # extract_wrong_data(root_path, out_path, "STR01")

    sample_root = r'D:\Working\Tianma\13902\data\0800\0825_add\13902\L3672B0108EB00'
    reconstruct_dataset(sample_root)