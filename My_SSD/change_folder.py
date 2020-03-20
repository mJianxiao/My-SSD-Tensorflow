import os
import os.path
from xml.etree.ElementTree import parse, Element




def test():
    path = "./VOC2007/Annotations/"
    files = os.listdir(path)  # 得到文件夹下所有文件名称
    s = []
    for xmlFile in files:  # 遍历文件夹
        if not os.path.isdir(xmlFile):  # 判断是否是文件夹,不是文件夹才打开
            print(xmlFile)
            pass
        path = "./VOC2007/Annotations/"
        newStr = os.path.join(path, xmlFile)
        dom = parse(newStr)  ###最核心的部分,路径拼接,输入的是具体路径
        root = dom.getroot()
        # print root
        #part = xmlFile[0:6]
        #part1 = part + '.jpg'
        newStr1 = 'JPEGImages'#图片的文件夹名称为JPEGImages

        # root.remove(root.find('path'))

        # e=Element('path')
        # print root.find('path').text
        root.find('folder').text = newStr1
        # print newStr
        # path[0]=newStr
        # n1[0:57]="E:/Project_dir/yolo_v2/darknet/build/darknet/x64/img/"
        # p0.firstChild.data='ok'
        # #打印输出
        print('folder after change')
        # print n0.firstChild.data

        # print '修改后的 pose'
        # print p0.firstChild.data
        # print '~~~~~'
        dom.write(newStr, xml_declaration=True)
        pass


if __name__ == '__main__':
    test()