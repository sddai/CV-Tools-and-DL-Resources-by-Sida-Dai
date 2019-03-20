# -*- coding: utf-8 -*-
# 第一种用法（替换已经复制了的文件），配置好下面要扫描替换的目录，直接运行 python copyNewToDest.py
# 第二种用法（直接复制文件），配置好SOURCE_PATH2（源目录）和DEST_PATH（目标目录），
	#运行 python copyNewToDest.py copyMobile, copyMobile中放目录中要copy的文件路径，多个文件换行即可
	#除了Application和Public文件夹，其它均需要正确的相对路径
# 第三种用法（配置多个copy目录），配置好对应条件后的代码，运行python copyNewToDest.py copyEasy easy, 最后一个参数值easy代表不同的项目
import os
import shutil
import sys

SOURCE_PATH1=r'C:\Users\jim\Desktop\tzb_mobile\0-wait-to-online';#比对目录，提供文件名
SOURCE_PATH2=r'D:\wamp\apps\mobile';#要复制出来的目录 D:\wamp\apps\mobile
DEST_PATH=r'C:\Users\jim\Desktop\tzb_mobile\0-wait-to-online'#目标目录

if len(sys.argv)==3 and sys.argv[2]=='easy':
	SOURCE_PATH1=r'C:\Users\jim\Desktop\tzb_easypro\0-wait-to-online';#比对目录，提供文件名
	SOURCE_PATH2=r'D:\wamp\apps\tzb_easy';#要复制出来的目录 D:\wamp\apps\mobile
	DEST_PATH=r'C:\Users\jim\Desktop\tzb_easypro\0-wait-to-online'#目标目录

def ReadFileNames(rootDir):
    FileList = []
    for parent,dirNames,fileNames in os.walk(rootDir):
        if fileNames:
            for fileName in fileNames:
                FileList.append(os.path.join(parent,fileName))
    return FileList

#搜索字符串中所有的子字符串，暂未用到
def SearchStr(str,search):
	start = 0
	while True:
		index = str.find(search, start)
		# if search string not found, find() returns -1
		# search is complete, break out of the while loop
		if index == -1:
			break
		print( "%s found at index %d" % (search, index) )
		# move to next possible start position
		start = index + 1

#main()
if len(sys.argv)==1:
	print sys.argv[0]
	if __name__=='__main__':
		if not os.path.exists(DEST_PATH):
			os.makedirs(DEST_PATH)
		fileList = ReadFileNames(SOURCE_PATH1)
		for oldfileName in fileList:
			print oldfileName
			newfileName = oldfileName.replace(SOURCE_PATH1, SOURCE_PATH2)
			destfileName = oldfileName.replace(SOURCE_PATH1, DEST_PATH)
			destdirName =  os.path.dirname(destfileName)
			if not os.path.exists(destdirName):
				os.makedirs(destdirName)
			shutil.copyfile(newfileName,destfileName)
			print 'copy',newfileName,'-->',destfileName
if len(sys.argv)>2 and len(sys.argv[1])>=4:
	if __name__=='__main__':
		print sys.argv[1]
		if not os.path.exists(DEST_PATH):
			os.makedirs(DEST_PATH)
		fileList = file(sys.argv[1]).readlines()
		for oldfileName in fileList:
			oldfileName=oldfileName.strip('\n')
			nPos = oldfileName.find('Application',0)
			if nPos<0:
				nPos = oldfileName.find('Public',0)
			if nPos>0:
				oldfileName = oldfileName[nPos-1:]
			#newfileName = oldfileName.replace(sys.argv[1], SOURCE_PATH2)
			#destfileName = oldfileName.replace(sys.argv[1], DEST_PATH)
			newfileName = SOURCE_PATH2 + oldfileName
			destfileName = DEST_PATH + oldfileName
			destdirName =  os.path.dirname(destfileName)
			if not os.path.exists(destdirName):
				os.makedirs(destdirName)
			shutil.copyfile(newfileName,destfileName)
			print 'copy',newfileName,'-->',destfileName
