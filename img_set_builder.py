
import os
import shutil
from random import randint

def buildTestAndVal(src, destTrain, destVal):
	print("building training and testing img sets in " + destTrain + " and " + destVal)
	if os.path.exists(destTrain):
		print("deleting previously found train set")
		shutil.rmtree(destTrain)
		print("deleting previously found val set")

	if os.path.exists(destVal):
		shutil.rmtree(destVal)

	for dirname, dirnames, filenames in os.walk(src):

			for filename in filenames:

				if filename.endswith(".jpg") or filename.endswith(".png"):
					
					copyDir(os.path.abspath(dirname), (destTrain + dirname.split("/")[-1]).replace(" ", "_"))
					moveDir(destTrain + dirname.split("/")[-1].replace(" ", "_"), (destVal + dirname.split("/")[-1]).replace(" ", "_"))
					break
					
	print("trining and testing img sets succesfully build")
def copyDir(root_src_dir, root_dst_dir):

	for src_dir, dirs, files in os.walk(root_src_dir):
		dst_dir = src_dir.replace(root_src_dir, root_dst_dir, 1)
		if not os.path.exists(dst_dir):
		    os.makedirs(dst_dir)
		for file_ in files:
		    src_file = os.path.join(src_dir, file_)
		    dst_file = os.path.join(dst_dir, file_)
		    if os.path.exists(dst_file):
		        os.remove(dst_file)
		    shutil.copy(src_file, dst_dir)

def moveDir(root_src_dir, root_dst_dir):

	for src_dir, dirs, files in os.walk(root_src_dir):
		dst_dir = src_dir.replace(root_src_dir, root_dst_dir, 1)
		if not os.path.exists(dst_dir):
		    os.makedirs(dst_dir)
		for file_ in files:
			if randint(1, 5) == 1: # 20%
			    src_file = os.path.join(src_dir, file_)
			    dst_file = os.path.join(dst_dir, file_)
			    if os.path.exists(dst_file):
			        os.remove(dst_file)
			    shutil.move(src_file, dst_dir)
							
					