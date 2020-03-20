# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 12:03:37 2020

@author: 82418
"""
import os
from PIL import Image


def IsValidImage(img_path):
    """
    判断文件是否为有效（完整）的图片
    :param img_path:图片路径
    :return:True：有效 False：无效
    """
    bValid = True
    try:
        Image.open(img_path).verify()
    except:
        bValid = False
    return bValid


def transimg(img_path,i):
    """
    转换图片格式
    :param img_path:图片路径
    :return: True：成功 False：失败
    """
    if IsValidImage(img_path):
        try:
            #str = img_path.rsplit(".", 1)
            output_img_path ='D:/runxun/tuxiangjiance/laji_xiangfang/laji_bc0310/4C_407/'+ "laji_4C_407_bc" +str(i)+ ".jpg"#str[0][17:]
            print(output_img_path)
            im = Image.open(img_path)
            im.save(output_img_path)
            return True
        except:
            return False
    else:
        return False


if __name__ == '__main__':
    path='D:/runxun/tuxiangjiance/laji_xiangfang/laji_bc0310/4C_407/'
    filelist=os.listdir(path)
    i=1
    for file in filelist:
        img_path=path+file
        transimg(img_path,i)
        i+=1