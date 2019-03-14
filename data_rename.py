# 文件批量重命名

import os
import sys

# path = "../data/"
path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".")

def rename(path):
	count = 0
	ori_dir = path
	for files in os.listdir(ori_dir):
		# Pass when a file is a folder #
		old_dir = os.path.join(ori_dir, files)
		if os.path.isdir(old_dir):
			continue
		# Rename #
		file_name = os.path.splitext(files)[0]
		file_type = os.path.splitext(files)[1]
		new_dir = os.path.join(ori_dir, str(count)+file_type)
		os.rename(old_dir, new_dir)
		count += 1
		print('count:', count)
rename(path) 

