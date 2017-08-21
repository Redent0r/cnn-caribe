import cnn3_driver
from os import listdir
from os.path import isfile, join

IMG_DIR = 'caribe_test/'

images = [IMG_DIR + f for f in listdir(IMG_DIR) if isfile(join(IMG_DIR, f))]

print(images)

for img in images:
	print(img + ' : ' + cnn3_driver.predict(img))

