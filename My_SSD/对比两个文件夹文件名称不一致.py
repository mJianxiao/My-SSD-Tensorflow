import os

path1='D:/lajmy/335-2-18/'
path2='D:/lajmy/335-2-18-label/'


list1=os.listdir(path1)
list11=[]
for file in list1:
    file = file[:len(file)-4]
    list11.append(file)

list22=[]
list2=os.listdir(path2)
for file in list2:
    file = file[:len(file)-4]
    list22.append(file)

for list_file1 in list11:
    if list_file1 in list22:
        pass
    else:
        print('list_file1 not in list22:',list_file1)
        os.remove(path1+list_file1+'.jpg')

for list_file2 in list22:
    if list_file2 in list11:
        pass
    else:
        print('list_file2 not in list11:',list_file2)
        os.remove(path2+list_file2+'.xml')


