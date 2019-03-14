# coding=utf-8
import os
import numpy as np
from sklearn.model_selection import train_test_split


def makeFileLists(imgPath, fileName='list.txt', withLabel=False, ext=['jpg','bmp','png']):
    '''
        makeFileList 函数用于包含多层目录的文件列表创建
        Params:
            imgPath     ：最上层的目录路径
            fileName    : 列表文件名称
            withLabel   : 默认为`False`,如果需要为每个图片路劲添加label,
                          则将该参数设置为`True`,图片所在目录的名称即为
                          该图片的label
            ext         : 图片格式
        Usage:
            makeFileLists('imagesRootPath', 'imageList.txt', False)
    '''
    # 判断路径是否存在
    if not os.path.exists(imgPath):
        print imagesPath, 'IS NOT EXIST, PLEASE CHECK IT!'

    # 判断路径是否为目录，如果是，则对目录下的子目录做递归操作
    elif os.path.isdir(imgPath):
        subPath = os.listdir(imgPath)  #相当于ls命令
        subPath = [os.path.join(imgPath,path) for path in subPath]
        for path in subPath:
            makeFileLists(path, fileName, withLabel)
    # 如果路径不是目录，则为图片的相对路径
    else:
        # 只保存指定格式的图片
        if imgPath[-3:] in ext:
            # 以追加的方式打开文件
            f = open(fileName,'a')
            # 如果需要添加label,则将图片所在目录的名称作为label
            if withLabel:
                line = imgPath+' '+(imgPath.split('/'))[-2]+'\n'
            else:
                line = imgPath+'\n'
            # 写入文件
            f.writelines(line)
            f.close()


def split(testsize=0.2):
    with open("list.txt", "rb") as f:
        lines = f.read().split('\n')
        lines = np.array(lines)
        # print lines
        # print  lines
        train, test = train_test_split(lines, test_size = testsize, random_state=1)
        # print 'train'
        # print train[4501]
        # print train[4502]
        # print "##################"
        # print train[7293]
        f2 = open('train_list.txt','a')
        for line in train:
            if line:    
                line = line + '\n'  # 跳过list.txt中的空行（list.txt的最后一行有一个换行符，有一个多余的空行）
                f2.writelines(line)
        f2.close()

        f3 = open('test_list.txt','a')
        for line in test:
            if line:
                line = line + '\n'
                f3.writelines(line)
        f3.close()


if __name__ == "__main__":
    imagesPath = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".")
    print imagesPath
    fileName = 'list.txt'
    if os.path.exists('list.txt'):
        os.remove('list.txt')
    makeFileLists(imagesPath, fileName, True)
    if os.path.exists('train_list.txt'):
        os.remove('train_list.txt')
    if os.path.exists('test_list.txt'):
        os.remove('test_list.txt')
    split(testsize=0.2)
