import shutil
import os
import glob


#当前路径文件拷贝到指定位置，可以指定拷贝数目
src = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".")
dst1 = '/home/vision4/VISION/daisida/data/blur/1/'
dst2 = '/home/vision4/VISION/daisida/data/blur/test/1/'

src += '/result/*.jpg'
src_paths = glob.glob(src)
result_paths = "./result/"

for src_index in range(len(src_paths)):
    if src_index <= 40000:
        shutil.copy(src_paths[src_index], dst1)
    else:
        shutil.copy(src_paths[src_index], dst2)
