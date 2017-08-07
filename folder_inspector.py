import os
from os import listdir
from os.path import isfile, join

def numberOfImages(dir):

	counter = 0
	for dirname, dirnames, filenames in os.walk(dir):
				for filename in filenames:
							counter += 1			
	return(counter)

def numberOfClasses(dir):
	return len(os.listdir(dir))

def getClassDict(classDir):
	classDict = {}
	counter = 0
	classes = os.listdir(classDir)
	for c in classes:
		classDict.update({counter:c})
		counter += 1

	return classDict

	

if __name__ == '__main__':
	print("number of images in caribe_train/: " + str(numberOfImages("caribe_train/")))
	print("number of images in caribe_val/: " + str(numberOfImages("caribe_val/")))
	print("number of classes in caribe_train/: " + str(numberOfClasses("caribe_train/")))
