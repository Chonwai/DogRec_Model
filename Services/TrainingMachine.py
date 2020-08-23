import keras
import tensorflowjs as tfjs
import numpy as np
import matplotlib.pyplot as plt
from Services.Utils import Utils
from Services.WriteResultServices import WriteResultServices

from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.densenet import DenseNet201, DenseNet169, DenseNet121
from keras.applications.nasnet import NASNetMobile
from keras.models import Sequential, Model
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D, GlobalAveragePooling1D
from keras import optimizers
from keras import metrics
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import *
from keras.utils import multi_gpu_model


class TrainingMachine:
    def __init__(self, classN=120):
        self.utils = Utils()
        self.trainX = []
        self.trainY = []
        self.testX = []
        self.testY = []
        self.classN = classN
        self.model = None
        self.parallelModel = None
        self.reduceLR = ReduceLROnPlateau(monitor='val_loss',
                                          factor=0.001,
                                          patience=20,
                                          verbose=1,
                                          mode='auto',
                                          min_delta=.0001,
                                          cooldown=0,
                                          min_lr=0.000001)
        self.earlyStop = EarlyStopping(monitor='val_loss',
                                       min_delta=.0002,
                                       patience=30,
                                       verbose=1,
                                       mode='auto',
                                       baseline=None,
                                       restore_best_weights=True)
        self.lambdaCallback = LambdaCallback(on_epoch_begin=self.epochBegin)
        self.dataGen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, featurewise_center=True, featurewise_std_normalization=True, rotation_range=15, rescale=1./255, shear_range=0.1, zoom_range=0.2, horizontal_flip=True, channel_shift_range=50)

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

    def top3Accuracy(self, y_true, y_pred):
        return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)
        self.__name__ = "top3Accuracy"
    
    def top5Accuracy(self, y_true, y_pred):
        return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)
        self.__name__ = "top5Accuracy"

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

    def epochBegin(self, epoch, logs):
        print("Start of " + str(epoch) + " Epoch. Learning Rate: " +
              str(K.eval(self.model.optimizer.lr)))

    def trainTFVGG19(self, ep=10, batch=15):
        pretrainingModel = VGG19(
            weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        self.model = Sequential()

        # for layer in pretrainingModel.layers:
        #     self.model.add(layer)

        # for layer in self.model.layers:
        #     layer.trainable = False
    
        self.model.add(pretrainingModel)

        self.model.add(GlobalAveragePooling2D(
            input_shape=self.trainX.shape[1:]))
        # self.model.add(Flatten(input_shape=self.trainX.shape[1:]))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(384, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.classN, activation='softmax'))

        self.model.summary()

        self.model.compile(optimizer=optimizers.RMSprop(
            lr=0.0001), loss='categorical_crossentropy', metrics=['mse', 'accuracy', self.top3Accuracy, self.top5Accuracy])

        history = self.model.fit(self.trainX, self.trainY, epochs=ep, batch_size=batch, validation_data=(
            self.testX, self.testY), callbacks=[self.earlyStop, self.lambdaCallback])

        self.model.save_weights('Model/VGG19_Dropout05_512_384_512.h5')

        self.topKAccuracy(self.model, k=1, testX=self.testX, testY=self.testY)
        self.topKAccuracy(self.model, k=3, testX=self.testX, testY=self.testY)
        self.topKAccuracy(self.model, k=5, testX=self.testX, testY=self.testY)
        WriteResultServices.create(history, 'TF_ImageNet_VGG19_Dropout04_512_384_512')
        self.utils.showReport(history)

    def trainTFVGG16(self, ep=10, batch=15):
        pretrainingModel = VGG16(
            weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        self.model = Sequential()
    
        self.model.add(pretrainingModel)

        self.model.add(GlobalAveragePooling2D(
            input_shape=self.trainX.shape[1:]))
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(2048, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(self.classN, activation='softmax'))

        self.model.summary()

        self.model.compile(optimizer=optimizers.RMSprop(
            lr=0.0001), loss='categorical_crossentropy', metrics=['mse', 'accuracy', self.top3Accuracy, self.top5Accuracy])

        history = self.model.fit(self.trainX, self.trainY, epochs=ep, batch_size=batch, validation_data=(
            self.testX, self.testY), callbacks=[self.reduceLR, self.lambdaCallback])

        self.model.save_weights('Model/TF_VGG16_200a.h5')

        self.topKAccuracy(self.model, k=1, testX=self.testX, testY=self.testY)
        self.topKAccuracy(self.model, k=3, testX=self.testX, testY=self.testY)
        self.topKAccuracy(self.model, k=5, testX=self.testX, testY=self.testY)
        self.utils.showReport(history)

    def trainTFMobileNetV2(self, ep=10, batch=15):
        pretrainingModel = MobileNetV2(
            weights=None, include_top=False, input_shape=(224, 224, 3))

        self.model = Sequential()

        self.model.add(pretrainingModel)

        self.model.add(GlobalAveragePooling2D(input_shape=self.trainX.shape[1:]))
        self.model.add(Dense(1280, activation='relu'))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(960, activation='relu'))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(1280, activation='relu'))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(self.classN, activation='softmax'))

        self.model.summary()

        self.model.compile(optimizer=optimizers.Adam(
            lr=0.0001), loss='categorical_crossentropy', metrics=['mse', 'accuracy', self.top3Accuracy, self.top5Accuracy])

        history = self.model.fit(self.trainX, self.trainY, epochs=ep, batch_size=batch, validation_data=(
            self.testX, self.testY), callbacks=[self.earlyStop, self.lambdaCallback])

        self.model.save_weights('Model/MobileNetV2_Dropout04_1280_960_1280_Adam_LR0001.h5')

        self.topKAccuracy(self.model, k=1, testX=self.testX, testY=self.testY)
        self.topKAccuracy(self.model, k=3, testX=self.testX, testY=self.testY)
        self.topKAccuracy(self.model, k=5, testX=self.testX, testY=self.testY)
        WriteResultServices.create(history, 'MobileNetV2_Dropout04_1280_960_1280_Adam_LR0001')
        self.utils.showReport(history)

    def trainTFXception(self, ep=10, batch=10):
        pretrainingModel = Xception(include_top=False, weights='imagenet', input_shape=(299, 299, 3))

        self.model = Sequential()

        self.model.add(pretrainingModel)

        self.model.add(GlobalAveragePooling2D(input_shape=self.trainX.shape[1:]))
        self.model.add(Dense(2048, activation='relu'))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(2048, activation='relu'))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(self.classN, activation='softmax'))

        self.model.summary()

        self.model.compile(optimizer=optimizers.RMSprop(
            lr=0.000001), loss='categorical_crossentropy', metrics=['mse', 'accuracy', self.top3Accuracy, self.top5Accuracy])

        history = self.model.fit(self.trainX, self.trainY, epochs=ep, batch_size=batch, validation_data=(
            self.testX, self.testY), callbacks=[self.earlyStop, self.lambdaCallback])

        self.model.save_weights('Model/TF_Xception_Dropout04_2048_512_2048_RMSprop_LR000001.h5')

        self.topKAccuracy(self.model, k=1, testX=self.testX, testY=self.testY)
        self.topKAccuracy(self.model, k=3, testX=self.testX, testY=self.testY)
        self.topKAccuracy(self.model, k=5, testX=self.testX, testY=self.testY)
        WriteResultServices.create(history, 'TF_Xception_Dropout04_2048_512_2048_RMSprop_LR000001')
        self.utils.showReport(history)
    
    def trainTFNASNetMobile(self, ep=10, batch=10):
        pretrainingModel = NASNetMobile(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

        self.model = Sequential()

        self.model.add(pretrainingModel)

        self.model.add(GlobalAveragePooling2D(input_shape=self.trainX.shape[1:]))
        self.model.add(Dense(1056, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(792, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1056, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.classN, activation='softmax'))

        self.model.summary()

        self.model.compile(optimizer=optimizers.RMSprop(
            lr=0.000001), loss='categorical_crossentropy', metrics=['mse', 'accuracy', self.top3Accuracy, self.top5Accuracy])

        history = self.model.fit(self.trainX, self.trainY, epochs=ep, batch_size=batch, validation_data=(
            self.testX, self.testY), callbacks=[self.earlyStop, self.lambdaCallback])

        self.model.save('Model/TF_NASNetMobile_Dropout05_1056_792_1056_RMSprop_LR000001.h5')

        self.topKAccuracy(self.model, k=1, testX=self.testX, testY=self.testY)
        self.topKAccuracy(self.model, k=3, testX=self.testX, testY=self.testY)
        self.topKAccuracy(self.model, k=5, testX=self.testX, testY=self.testY)
        WriteResultServices.create(history, 'TF_NASNetMobile_Dropout05_1056_792_1056_RMSprop_LR000001')
        self.utils.showReport(history)


    def trainTFInceptionResNetV2(self, ep=10, batch=15):
        pretrainingModel = InceptionResNetV2(
            weights='imagenet', include_top=False, input_shape=(299, 299, 3))

        self.model = Sequential()

        self.model.add(pretrainingModel)

        self.model.add(GlobalAveragePooling2D(input_shape=self.trainX.shape[1:]))
        self.model.add(Dense(1536, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1536, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1536, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.classN, activation='softmax'))

        self.model.summary()

        self.model.compile(optimizer=optimizers.RMSprop(
            lr=0.00001), loss='categorical_crossentropy', metrics=['mse', 'accuracy', self.top3Accuracy, self.top5Accuracy])

        history = self.model.fit(self.trainX, self.trainY, epochs=ep, batch_size=batch, validation_data=(
            self.testX, self.testY), callbacks=[self.reduceLR, self.lambdaCallback])

        self.model.save_weights('Model/TF_InceptionResNetV2_Dropout05_1536_1536_1536.h5')

        self.topKAccuracy(self.model, k=1, testX=self.testX, testY=self.testY)
        self.topKAccuracy(self.model, k=3, testX=self.testX, testY=self.testY)
        self.topKAccuracy(self.model, k=5, testX=self.testX, testY=self.testY)
        WriteResultServices.create(history, 'TF_InceptionResNetV2_Dropout05_1536_1536_1536')
        self.utils.showReport(history)

    def trainTFNASNetLarge(self, ep=10, batch=15):
        pretrainingModel = NASNetLarge(
            weights='imagenet', include_top=False, input_shape=(331, 331, 3))

        self.model = Sequential()

        pretrainingModel.trainable = False

        self.model.add(pretrainingModel)

        self.model.add(GlobalAveragePooling2D(input_shape=self.trainX.shape[1:]))
        self.model.add(Dense(4032, activation='relu'))
        self.model.add(Dropout(0.5))
        # self.model.add(Dense(2016, activation='relu'))
        # self.model.add(Dropout(0.5))
        self.model.add(Dense(4032, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.classN, activation='softmax'))

        self.model.summary()

        self.model.compile(optimizer=optimizers.RMSprop(
            lr=0.000001), loss='categorical_crossentropy', metrics=['mse', 'accuracy', self.top3Accuracy, self.top5Accuracy])

        history = self.model.fit(self.trainX, self.trainY, epochs=ep, batch_size=batch, validation_data=(
            self.testX, self.testY), callbacks=[self.earlyStop, self.lambdaCallback])

        self.model.save_weights('Model/TF_NASNetLarge_Dropout05_4032_4032_RMSprop_LR000001.h5')

        self.topKAccuracy(self.model, k=1, testX=self.testX, testY=self.testY)
        self.topKAccuracy(self.model, k=3, testX=self.testX, testY=self.testY)
        self.topKAccuracy(self.model, k=5, testX=self.testX, testY=self.testY)
        WriteResultServices.create(history, 'TF_NASNetLarge_Dropout05_4032_4032_RMSprop_LR000001')
        self.utils.showReport(history)
    
    def trainMobileNetV2(self, ep=10, batch=15):
        self.model = MobileNetV2(include_top=True, weights=None,
                            input_shape=(224, 224, 3),  classes=self.classN)

        self.model.summary()

        self.model.compile(optimizer=optimizers.RMSprop(
            lr=0.0001), loss='categorical_crossentropy', metrics=['mse', 'accuracy', self.top3Accuracy, self.top5Accuracy])

        history = self.model.fit(self.trainX, self.trainY, epochs=ep, batch_size=batch, validation_data=(
            self.testX, self.testY), callbacks=[self.earlyStop, self.lambdaCallback])

        self.model.save_weights('Model/MobileNetV2_LR0001.h5')

        self.topKAccuracy(self.model, k=1, testX=self.testX, testY=self.testY)
        self.topKAccuracy(self.model, k=3, testX=self.testX, testY=self.testY)
        self.topKAccuracy(self.model, k=5, testX=self.testX, testY=self.testY)
        WriteResultServices.create(history, 'MobileNetV2_LR0001')
        self.utils.showReport(history)

    def trainXception(self, ep=10, batch=15, pooling='avg'):
        self.model = Xception(include_top=True, weights=None,
                         input_shape=(299, 299, 3), classes=self.classN)

        self.model.summary()

        self.model.compile(optimizer=optimizers.RMSprop(
            lr=0.0001), loss='categorical_crossentropy', metrics=['mse', 'accuracy', self.top3Accuracy, self.top5Accuracy])

        history = self.model.fit(self.trainX, self.trainY, epochs=ep, batch_size=batch, validation_data=(
            self.testX, self.testY), callbacks=[self.earlyStop, self.lambdaCallback])

        self.model.save_weights('Model/Xception.h5')

        self.topKAccuracy(self.model, k=1, testX=self.testX, testY=self.testY)
        self.topKAccuracy(self.model, k=3, testX=self.testX, testY=self.testY)
        self.topKAccuracy(self.model, k=5, testX=self.testX, testY=self.testY)
        WriteResultServices.create(history, 'Xception')
        self.utils.showReport(history)

    def trainVGG19(self, ep=10, batch=15, pooling='avg'):
        self.model = VGG19(include_top=True, weights=None,
                      input_shape=(224, 224, 3), classes=self.classN)

        self.model.summary()

        self.model.compile(optimizer=optimizers.RMSprop(
            lr=0.0001), loss='categorical_crossentropy', metrics=['mse', 'accuracy', self.top3Accuracy, self.top5Accuracy])

        history = self.model.fit(self.trainX, self.trainY, epochs=ep, batch_size=batch, validation_data=(
            self.testX, self.testY), callbacks=[self.earlyStop, self.lambdaCallback])

        self.model.save_weights('Model/VGG19.h5')

        self.topKAccuracy(self.model, k=1, testX=self.testX, testY=self.testY)
        self.topKAccuracy(self.model, k=3, testX=self.testX, testY=self.testY)
        self.topKAccuracy(self.model, k=5, testX=self.testX, testY=self.testY)
        WriteResultServices.create(history, 'VGG19')
        self.utils.showReport(history)
    
    def trainNASNetLarge(self, ep=10, batch=15, pooling='avg'):
        self.model = NASNetLarge(include_top=True, weights=None,
                         input_shape=(331, 331, 3), classes=self.classN)

        self.model.summary()

        self.model.compile(optimizer=optimizers.RMSprop(
            lr=0.001), loss='categorical_crossentropy', metrics=['mse', 'accuracy', self.top3Accuracy, self.top5Accuracy])

        history = self.model.fit(self.trainX, self.trainY, epochs=ep, batch_size=batch, validation_data=(
            self.testX, self.testY), callbacks=[self.earlyStop, self.lambdaCallback])

        self.model.save_weights('Model/NASNetLarge_LR001.h5')

        self.topKAccuracy(self.model, k=1, testX=self.testX, testY=self.testY)
        self.topKAccuracy(self.model, k=3, testX=self.testX, testY=self.testY)
        self.topKAccuracy(self.model, k=5, testX=self.testX, testY=self.testY)
        WriteResultServices.create(history, 'NASNetLarge_LR001')
        self.utils.showReport(history)

    def eval(self, img, k=5):
        pretrainingModel = MobileNetV2(
            weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        self.model = Sequential()

        self.model.add(pretrainingModel)

        self.model.add(GlobalAveragePooling2D())
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.classN, activation='softmax'))

        self.model.load_weights('Model/TF_MobileNetV2_100a.h5')

        predict = self.model.predict(img)
        probability = self.model.predict_proba(img)[0]
        result = predict[0]
        topK = sorted(range(len(result)),
                      key=lambda i: result[i], reverse=True)[:k]
        topKPercentage = probability[topK[:k]]
        return topK, topKPercentage
