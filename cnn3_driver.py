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

#img_src = "Caribe/" # original
img_src = "Caribe_sub/" # subset
train_data_dir = 'caribe_train/'
validation_data_dir = 'caribe_val/'

epochs = 50
batch_size = 3

img_set_builder.buildTestAndVal(img_src, train_data_dir, validation_data_dir) # run once

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


#---------------------------------------------------------
# model = Sequential()
# model.add(Conv2D(32, (3, 3), input_shape=input_shape))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#----------------------------------------------------------

#Get back the convolutional part of a VGG network trained on ImageNet
model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
model_vgg16_conv.summary()

#Create your own input format (here 3x200x200)
input = Input(shape= input_shape,name = 'image_input')

#Use the generated model 
output_vgg16_conv = model_vgg16_conv(input)

#Add the fully-connected layers 
x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dense(numberOfClasses, activation='softmax', name='predictions')(x)

#Create your own model 
my_model = Model(input=input, output=x)

# model.add(Flatten())
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# #model.add(Dropout(0.25))
# model.add(Dense(numberOfClasses))
# model.add(Activation('sigmoid'))

my_model.compile(loss='categorical_crossentropy',
               optimizer='adam',
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