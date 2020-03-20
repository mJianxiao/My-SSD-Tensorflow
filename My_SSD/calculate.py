import re
import os
import xml.etree.ElementTree as ET

class1 = 'non-motor'
class2 = 'car'
class3 = 'person'
class4 = 'luandw'
class5 = 'lajmy'

annotation_folder = './VOC2007/Annotations/'  # 改为自己标签文件夹的路径
list = os.listdir(annotation_folder)


def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.xml':
                L.append(os.path.join(root, file))
    return L


total_number1 = 0
total_number2 = 0
total_number3 = 0
total_number4 = 0
total_number5 = 0
total = 0
total_pic = 0

pic_num1 = 0
pic_num2 = 0
pic_num3 = 0
pic_num4 = 0
pic_num5 = 0


flag1 = 0
flag2 = 0
flag3 = 0
flag4 = 0
flag5 = 0


xml_dirs = file_name(annotation_folder)

for i in range(0, len(xml_dirs)):
    print(xml_dirs[i])
    # path = os.path.join(annotation_folder,list[i])
    # print(path)

    annotation_file = open(xml_dirs[i]).read()

    root = ET.fromstring(annotation_file)
    # tree = ET.parse(annotation_file)
    # root = tree.getroot()

    total_pic = total_pic + 1
    for obj in root.findall('object'):
        label = obj.find('name').text
        if label == class1:
            total_number1 = total_number1 + 1
            flag1 = 1
            total = total + 1
        # print("bounding box number:", total_number1)
        if label == class2:
            total_number2 = total_number2 + 1
            flag2 = 1
            total = total + 1
        if label == class3:
            total_number3 = total_number3 + 1
            flag3 = 1
            total = total + 1
        if label == class4:
            total_number4 = total_number4 + 1
            flag4 = 1
            total = total + 1
        if label == class5:
            total_number5 = total_number5 + 1
            flag5 = 1
            total = total + 1


    if flag1 == 1:
        pic_num1 = pic_num1 + 1
        # print("pic number:", pic_num1)
        flag1 = 0
    if flag2 == 1:
        pic_num2 = pic_num2 + 1
        flag2 = 0
    if flag3 == 1:
        pic_num3 = pic_num3 + 1
        flag3 = 0
    if flag4 == 1:
        pic_num4 = pic_num4 + 1
        flag4 = 0
    if flag5 == 1:
        pic_num5 = pic_num5 + 1
        flag5 = 0


print(class1, pic_num1, total_number1)
print(class2, pic_num2, total_number2)
print(class3, pic_num3, total_number3)
print(class4, pic_num4, total_number4)
print(class5, pic_num5, total_number5)


print("total", total_pic, total)
