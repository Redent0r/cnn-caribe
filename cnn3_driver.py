from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
import numpy as np

import os
import shutil
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import folder_inspector
import img_set_builder

# dimensions of our images.
img_width, img_height = 150, 150

img_src = "Caribe/" # original
#img_src = "Caribe_sub/" # subset
train_data_dir = 'caribe_train/'
validation_data_dir = 'caribe_val/'

epochs = 50
batch_size = 32

#img_set_builder.buildTestAndVal(img_src, train_data_dir, validation_data_dir) # run once

nb_train_samples = folder_inspector.numberOfImages(train_data_dir)
nb_validation_samples = folder_inspector.numberOfImages(validation_data_dir)

print("number of train samples: " + str(nb_train_samples) + " in " + train_data_dir)
print("number of val samples: " + str(nb_validation_samples) + " in " + validation_data_dir)

numberOfClasses = folder_inspector.numberOfClasses(train_data_dir)
print("number of classes: " + str(numberOfClasses) + " in " + train_data_dir)

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height) # (depth, width, height)
else:
    input_shape = (img_width, img_height, 3)


#Create your own input format (here 3x200x200)
input = Input(shape= input_shape,name = 'image_input')

# build the VGG16 network
model = applications.VGG16(weights='imagenet', include_top=False)
model.summary()

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=input))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning

# add the model on top of the convolutional base
model.add(top_model)

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:25]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

my_model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    #validation_steps=1)
    validation_steps=nb_validation_samples // batch_size)

my_model.save('first_try3.h5')