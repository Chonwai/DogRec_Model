import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras
import numpy as np

from sklearn.model_selection import train_test_split
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
from keras import metrics


class TransferLearning:
    def __init__(self, x, y, classN):
        self.x = x
        self.y = y
        self.trainX = []
        self.trainY = []
        self.testX = []
        self.testY = []
        self.classN = classN

    def init(self):
        trainX, testX, trainY, testY = train_test_split(
            self.x, self.y, test_size=0.25, random_state=0)
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

    def trainVGG(self):
        vgg = VGG19(weights='imagenet', include_top=False,
                    input_shape=(224, 224, 3))

        model = Sequential()

        for layer in vgg.layers:
            model.add(layer)

        for layer in model.layers:
            layer.trainable = False

        model.add(Flatten(input_shape=self.trainX.shape[1:]))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.classN, activation='softmax'))

        model.summary()

        model.compile(optimizer=optimizers.RMSprop(
            lr=0.0001), loss='categorical_crossentropy', metrics=['mse', 'accuracy'])

        model.fit(self.trainX, self.trainY, epochs=30, batch_size=15,
                  validation_data=(self.testX, self.testY))

        model.save_weights('Model/TransferLearning.h5')
