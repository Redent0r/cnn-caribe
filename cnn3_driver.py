'''
Using Bottleneck Features for Multi-Class Classification in Keras
We use this technique to build powerful (high accuracy without overfitting) Image Classification systems with small
amount of training data.
The full tutorial to get this code working can be found at the "Codes of Interest" Brog at the following link,
http://www.codesofinterest.com/2017/08/bottleneck-features-multi-class-classification-keras.html
Please go through the tutorial before attrmpting to run this code, as it explains how to setup your training data.
The code was tested on Python 3.5, with the following library versions,
Keras 2.0.6
TensorFlow 1.2.1
OpenCV 3.2.0
This should work with Theano as well, but untested.

reference: https://gist.github.com/Thimira/354b90d59faf8b0d758f74eae3a511e2

VGG16 on Caribe sweetspot:
128 batch, 50 epock, without dropout
'''

import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import math
import cv2
import img_set_builder

img_src = "Caribe/" # original
#img_src = "Caribe_sub/" # subset
train_data_dir = 'caribe_train/'
validation_data_dir = 'caribe_val/'

#img_set_builder.buildTestAndVal(img_src, train_data_dir, validation_data_dir) # run once

# dimensions of our images.
img_width, img_height = 224, 224

top_model_weights_path = 'bottleneck_fc_model.h5'

# number of epochs to train top model
epochs = 50
# batch size used by flow_from_directory and predict_generator
batch_size = 128

# # with data augmentation
# datagen_train = ImageDataGenerator(rescale=1. / 255,
#     			shear_range = 0.2,
#     			zoom_range = 0.2,
#     			horizontal_flip = True)

# without data augmentation and horizontal flip
datagen_train = ImageDataGenerator(rescale=1. / 255,
                horizontal_flip = True)

datagen_val = ImageDataGenerator(rescale=1. / 255)

def save_bottlebeck_features():

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')
    #model = applications.resnet50.ResNet50(include_top=False, weights='imagenet')

    generator = datagen_train.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    print(len(generator.filenames))
    #print(generator.class_indices)
    print(len(generator.class_indices))

    nb_train_samples = len(generator.filenames)
    num_classes = len(generator.class_indices)

    predict_size_train = int(math.ceil(nb_train_samples / batch_size))

    bottleneck_features_train = model.predict_generator(
        generator, predict_size_train)

    np.save('bottleneck_features_train.npy', bottleneck_features_train)

    generator = datagen_val.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    nb_validation_samples = len(generator.filenames)

    predict_size_validation = int(
        math.ceil(nb_validation_samples / batch_size))

    bottleneck_features_validation = model.predict_generator(
        generator, predict_size_validation)

    np.save('bottleneck_features_validation.npy',
            bottleneck_features_validation)

def train_top_model():

    generator_top = datagen_train.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    nb_train_samples = len(generator_top.filenames)
    num_classes = len(generator_top.class_indices)

    # save the class indices to use use later in predictions
    np.save('class_indices.npy', generator_top.class_indices)

    # load the bottleneck features saved earlier
    train_data = np.load('bottleneck_features_train.npy')

    # get the class lebels for the training data, in the original order
    train_labels = generator_top.classes

    # https://github.com/fchollet/keras/issues/3467
    # convert the training labels to categorical vectors
    train_labels = to_categorical(train_labels, num_classes=num_classes)

    generator_top = datagen_val.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    nb_validation_samples = len(generator_top.filenames)

    validation_data = np.load('bottleneck_features_validation.npy')

    validation_labels = generator_top.classes
    validation_labels = to_categorical(
        validation_labels, num_classes=num_classes)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))

    # setting clipnorm and clipvalue helps with loss = nan but does not solves it
    rmsOpt = keras.optimizers.RMSprop(lr=2e-4, rho=0.9, epsilon=1e-08, decay=0.0, clipnorm=1. , clipvalue=0.5)

    model.compile(optimizer=rmsOpt,
                  loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, train_labels,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(validation_data, validation_labels))

    model.save_weights(top_model_weights_path)

    (eval_loss, eval_accuracy) = model.evaluate(
        validation_data, validation_labels, batch_size=batch_size, verbose=1)

    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
    print("[INFO] Loss: {}".format(eval_loss))

    plt.figure(1)

    # summarize history for accuracy

    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss

    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def predict(image_path, top_results):
    # load the class_indices saved in the earlier step
    class_dictionary = np.load('class_indices.npy').item()

    num_classes = len(class_dictionary)

    # add the path to your test image below

    orig = cv2.imread(image_path)

    #print("[INFO] loading and preprocessing image...")
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)

    # important! otherwise the predictions will be '0'
    image = image / 255

    image = np.expand_dims(image, axis=0)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    # get the bottleneck prediction from the pre-trained VGG16 model
    bottleneck_prediction = model.predict(image)

    # build top model
    model = Sequential()
    model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))

    model.load_weights(top_model_weights_path)

    # use the bottleneck prediction on the top model to get the final
    # classification
    class_predicted = model.predict_classes(bottleneck_prediction, verbose=0)

    #print('class predicted: ')
    #print(class_predicted)


    probabilities = model.predict_proba(bottleneck_prediction, verbose=0).tolist()[0]


    #print('prob: ')
    #print(probabilities)

    topX = sorted(zip(probabilities), reverse=True)[:top_results]
    #print(type(topX))

    label = []

    inID = class_predicted[0]

    inv_map = {v: k for k, v in class_dictionary.items()}

    #label = inv_map[inID]

    for prob in topX:
        label.append(inv_map[probabilities.index(prob[0])])


    # get the prediction label
    #print("Image ID: {}, Label: {}".format(inID, label))
    return label

    # # display the predictions with the image
    # cv2.putText(orig, "Predicted: {}".format(label), (10, 30),
    #             cv2.FONT_HERSHEY_PLAIN, 1.5, (43, 99, 255), 2)

    # cv2.imshow("Classification", orig)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":

	save_bottlebeck_features()
	train_top_model()
	#predict()
	#cv2.destroyAllWindows()