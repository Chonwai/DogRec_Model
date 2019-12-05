import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras
import numpy as np
import matplotlib.pyplot as plt
from Services.Utils import Utils

from sklearn.model_selection import train_test_split
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenetv2 import MobileNetV2
from keras.models import Sequential, Model
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D, GlobalAveragePooling1D
from keras import optimizers
from keras import metrics
from keras.callbacks import *


class TrainingMachine:
    def __init__(self, classN):
        self.utils = Utils()
        self.trainX = []
        self.trainY = []
        self.testX = []
        self.testY = []
        self.classN = classN
        self.reducelr = ReduceLROnPlateau(monitor='val_acc',
                                          factor=np.sqrt(.1),
                                          patience=4,
                                          verbose=1,
                                          mode='auto',
                                          min_delta=.0001,
                                          cooldown=0,
                                          min_lr=0.00000001)

    def init(self, x, y):
        trainX, testX, trainY, testY = train_test_split(
            x, y, test_size=0.3, random_state=0)
        self.convertNpArray(trainX, testX, trainY, testY)

    def convertNpArray(self, trainX, testX, trainY, testY):
        for i in range(0, len(trainY)):
            data = np.array(trainX[i])
            label = np.array(trainY[i])
            self.trainX.append(data)
            self.trainY.append(label)

        for i in range(0, len(testY)):
            data = np.array(testX[i])
            label = np.array(testY[i])
            self.testX.append(data)
            self.testY.append(label)

        self.trainX = np.array(self.trainX)
        self.trainY = np.array(self.trainY)
        self.testX = np.array(self.testX)
        self.testY = np.array(self.testY)

    def topKAccuracy(self, model, k=3, testX=[], testY=[]):
        predict = model.predict_proba(testX)
        count = 0
        for i in range(len(predict)):
            topK = sorted(
                range(len(predict[i])), key=lambda x: predict[i][x], reverse=True)[:k]
            for j in topK:
                if (j == np.argmax(testY[i])):
                    count = count + 1
        meanAccuracy = (count / len(predict)) * 100
        print("Top " + str(k) + " Accuracy is: " + str(meanAccuracy))

    def trainTFVGG19(self, ep=10, batch=15):
        pretrainingModel = VGG19(
            weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        model = Sequential()

        for layer in pretrainingModel.layers:
            model.add(layer)

        for layer in model.layers:
            layer.trainable = False

        model.add(GlobalAveragePooling2D(input_shape=self.trainX.shape[1:]))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(self.classN, activation='softmax'))

        model.summary()

        model.compile(optimizer=optimizers.Adam(
            lr=0.001), loss='categorical_crossentropy', metrics=['mse', 'accuracy'])

        history = model.fit(self.trainX, self.trainY, epochs=ep, validation_data=(
            self.testX, self.testY), callbacks=[self.reducelr])

        model.save_weights('Model/TF_VGG19_100.h5')

        self.topKAccuracy(model, k=1, testX=self.testX, testY=self.testY)
        self.topKAccuracy(model, k=3, testX=self.testX, testY=self.testY)
        self.topKAccuracy(model, k=5, testX=self.testX, testY=self.testY)
        self.utils.showReport(history)

    def trainTFMobileNetV2(self, ep=10, batch=15):
        pretrainingModel = MobileNetV2(
            weights='imagenet', include_top=False, input_shape=(160, 160, 3))

        model = Sequential()

        # for layer in pretrainingModel.layers:
        #     model.add(layer)

        # for layer in model.layers:
        #     layer.trainable = False

        # pretrainingModel.trainable = False

        model.add(pretrainingModel)

        model.add(GlobalAveragePooling2D(input_shape=self.trainX.shape[1:]))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.classN, activation='softmax'))

        model.summary()

        model.compile(optimizer=optimizers.RMSprop(
            lr=0.001), loss='categorical_crossentropy', metrics=['mse', 'accuracy'])

        model.fit(self.trainX, self.trainY, epochs=ep,
                  validation_data=(self.testX, self.testY))

        model.save_weights('Model/TF_MobileNetV2_01.h5')

    def trainMobileNetV2(self, ep=10, batch=15):

        model = MobileNetV2(include_top=True, weights=None,
                            input_shape=(168, 168, 3),  classes=self.classN)

        model.summary()

        model.compile(optimizer=optimizers.RMSprop(
            lr=0.0001), loss='categorical_crossentropy', metrics=['mse', 'accuracy'])

        model.fit(self.trainX, self.trainY, epochs=ep,
                  batch_size=batch, validation_data=(self.testX, self.testY))

        model.save_weights('Model/MobileNetV2_01.h5')

    def trainXception(self, ep=10, batch=15, pooling='avg'):
        model = Xception(include_top=True, weights=None,
                         input_shape=(75, 75, 3), classes=self.classN)

        model.summary()

        model.compile(optimizer=optimizers.RMSprop(
            lr=0.001), loss='categorical_crossentropy', metrics=['mse', 'accuracy'])

        model.fit(self.trainX, self.trainY, epochs=ep,
                  batch_size=batch, validation_data=(self.testX, self.testY))

        model.save_weights('Model/20191121_MobileNetV2.h5')

    def trainVGG19(self, ep=10, batch=15, pooling='avg'):
        model = VGG19(include_top=True, weights=None,
                      input_shape=(75, 75, 3), classes=self.classN)

        model.summary()

        model.compile(optimizer=optimizers.RMSprop(
            lr=0.001), loss='categorical_crossentropy', metrics=['mse', 'accuracy'])

        model.fit(self.trainX, self.trainY, epochs=ep,
                  batch_size=batch, validation_data=(self.testX, self.testY))

        model.save_weights('Model/20191121_MobileNetV2.h5')

    def eval(self, img, k=5):
        pretrainingModel = VGG19(
            weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        model = Sequential()

        for layer in pretrainingModel.layers:
            model.add(layer)

        model.add(GlobalAveragePooling2D(input_shape=img.shape[1:]))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(self.classN, activation='softmax'))

        model.load_weights('Model/TF_VGG19_06.h5')

        predict = model.predict(img)
        result = predict[0]
        topK = sorted(range(len(result)),
                      key=lambda i: result[i], reverse=True)[:k]
        print(topK)