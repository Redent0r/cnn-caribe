import os
import shutil
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import folder_inspector
import img_set_builder

#from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# ### data transformation preview ###
# datagen = ImageDataGenerator( # data generator
#         rotation_range=40,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#        fill_mode='nearest')

# img = load_img('caribe_train/Acanthurus_chirurgus/Acanthurus chirurgus [Robertson & Van Tassell [National aquaria [NC ]176.jpg')
# x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
# x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# if os.path.exists("preview/"):
#   shutil.rmtree("preview/")
# os.mkdir("preview")

# i = 0
# for batch in datagen.flow(x, batch_size=1,
#                           save_to_dir='preview/', save_prefix='Acantharus_cirugus', save_format='jpeg'):
#     i += 1
#     if i == 20:
#         break  # otherwise the generator would loop indefinitely

# dimensions of our images.
img_width, img_height = 150, 150

#img_src = "Caribe/" # original
img_src = "Caribe_sub/" # subset
train_data_dir = 'caribe_train/'
validation_data_dir = 'caribe_val/'

epochs = 50
batch_size = 2

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

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
#---------------------------------------------------
model.add(Conv2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
#model.add(Dropout(0.5))
 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(numberOfClasses, activation='softmax'))
#-----------------------------------------------------
model.compile(loss='categorical_crossentropy',
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

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    #validation_steps=1)
    validation_steps=nb_validation_samples // batch_size)

model.save('first_try2.h5')

