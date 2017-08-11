# https://gist.github.com/fchollet/f35fbc80e066a49d65f1688a7e99f069
import os
import shutil
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import folder_inspector
import img_set_builder
from keras import optimizers
from keras import applications
import numpy as np

# dimensions of our images.
img_width, img_height = 150, 150

top_model_weights_path = 'bottleneck_fc_model.h5'

#img_src = "Caribe/" # original
img_src = "Caribe_sub/" # subset

train_data_dir = 'caribe_train/'
validation_data_dir = 'caribe_val/'

#img_set_builder.buildTestAndVal(img_src, train_data_dir, validation_data_dir) # run once

nb_train_samples = folder_inspector.numberOfImages(train_data_dir)
nb_validation_samples = folder_inspector.numberOfImages(validation_data_dir)
numberOfClasses = folder_inspector.numberOfClasses(train_data_dir)

epochs = 50
batch_size = 16

train_steps = nb_train_samples // batch_size if nb_train_samples // batch_size  != 0 else 1
val_steps = nb_validation_samples // batch_size if nb_validation_samples // batch_size != 0 else 1

def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    train_steps = nb_train_samples // batch_size if nb_train_samples // batch_size  != 0 else 1
    val_steps = nb_validation_samples // batch_size if nb_validation_samples // batch_size != 0 else 1

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, train_steps)
    np.save(open('bottleneck_features_train.npy', 'wb'),
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, val_steps)
    np.save(open('bottleneck_features_validation.npy', 'wb'),
            bottleneck_features_validation)

def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
    train_labels = np.array(
        [0] * (nb_train_samples // 2) + [1] * (nb_train_samples // 2))

    validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
    validation_labels = np.array(
        [0] * (nb_validation_samples // 2) + [1] * (nb_validation_samples // 2))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(numberOfClasses, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)

save_bottlebeck_features()
train_top_model()



