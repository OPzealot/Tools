#!/usr/bin/env python
# encoding:utf-8
"""
author: liusili
@contact: liusili@unionbigdata.com
@software:
@file: generate_xml
@time: 2020/3/27
@desc: 
"""
import os
from xml.dom.minidom import Document


def generate_xml(image_info, category, bbox_list, out_dir=None):
    """
    :param out_dir:
    :info: 输出xml文件
    :param image_info:
    :param category:
    :param bbox_list:
    :return:
    """
    img_name = image_info['file_name']
    img_h = str(image_info['height'])
    img_w = str(image_info['width'])
    img_path = image_info['path']

    doc = Document()
    orderpack = doc.createElement("annotation")
    doc.appendChild(orderpack)

    objectfolder = doc.createElement("folder")
    objectcontenttext = doc.createTextNode(category)
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

    for bbox in bbox_list:
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[0] + bbox[2]
        ymax = bbox[1] + bbox[3]

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


if __name__ == '__main__':
    from tqdm import tqdm
    import sys
    from findROI import FindROI

    root_path = r'D:\Working\Tianma\13902\TEST\13902_testset'
    bbox_lst = [[0, 0, 1, 1]]
    pbar = tqdm(os.listdir(root_path), file=sys.stdout)
    for category in pbar:
        category_path = os.path.join(root_path, category)
        for file in os.listdir(category_path):
            if os.path.splitext(file)[-1] != '.jpg':
                continue
            tqdm.write('[START] 开始打标： {}'.format(file))
            image_info = {}
            image_info['file_name'] = file

            img_path = os.path.join(category_path, file)
            image_info['path'] = img_path
            try:
                roi = FindROI(img_path)
            except: continue
            image_info['height'] = roi.height
            image_info['width'] = roi.width
            generate_xml(image_info, category, bbox_lst)

    print('Finish.')

