from PIL import Image
import os.path
import glob
import argparse

def convertjpg(jpgfile,outdir,width=224,height=224):
	img = Image.open(jpgfile)
	try:
		new_img = img.resize((width, height), Image.BILINEAR)
		new_img.save(os.path.join(outdir,os.path.basename(jpgfile)))
	except Exception as e:
		print(e)
				
parser = argparse.ArgumentParser(description='resize pictures')
parser.add_argument('-dir', type=str, default='/home/data/testset', help='Pass a directory to resize the images in it')
parser.add_argument('-width', type=str, default='/home/data/testset', help='Pass a directory to resize the images in it')
parser.add_argument('-height', type=str, default='/home/data/testset', help='Pass a directory to resize the images in it')

for jpgfile in glob.glob("/home/data/testset/*.jpg"):
	try:
		convertjpg(jpgfile,"/homedata/testset/resize")
	except Exception as e:
		print(e)
