
import os


tihuan_path='D:/runxun/tuxiangjiance/laji_xiangfang/laji_bc0310/'

filelist_tihuan = os.listdir('D:/runxun/tuxiangjiance/laji_xiangfang/laji_bc0310/4C_407/')
uselist = os.listdir('D:/runxun/tuxiangjiance/laji_xiangfang/laji_bc0310/4C_407_label/')
i = 0
for file in filelist_tihuan:
    used_name = tihuan_path +'4C_407_label/'+ uselist[i]
    file_name = tihuan_path +'4C_407_label/'+ filelist_tihuan[i][:len(filelist_tihuan[i])-4] + '.xml'
    os.rename(used_name, file_name)
    i +=1


'''
# -*- coding: UTF-8 -*-
import os

#获得文件夹下文件名列表
path=r"G:\BaiduNetdiskDownload\第1册"
path=unicode(path,"utf8")
file_list=os.listdir(path)

#选择要重命名的文件夹路径
os.chdir(path)

#将文件名中的Lesson和空格用空字符串替代
for file in file_list:
    os.rename(file,file.replace("Lesson ",""))
'''
