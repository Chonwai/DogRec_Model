import os
import matplotlib.pyplot as plt
from PIL import Image
import skimage.io
import skimage.transform
import xml.etree.ElementTree as ET
import numpy as np

class Utils():
    def __init__(self, N=120, datasetPath='./Data/Training/', annotationPath='./Data/Annotation/'):
        self.datasetPath = datasetPath
        self.annotationPath = annotationPath
        self.folderList = []
        self.N = N

    def init(self):
        self.folderList = []
        if (self.N > 120):
            self.N = 120

    def readDatasetFolderName(self):
        root = os.listdir(self.datasetPath)
        for i in root:
            if (i == '.DS_Store'):
                continue
            self.folderList.append(i)
        self.folderList.sort()

    def loadDataset(self):
        images = []
        labels = []
        self.readDatasetFolderName()
        count = 0
        for i in self.folderList[0:self.N]:
            imgList = os.listdir(self.datasetPath + '/' + i)
            annotationList = os.listdir(self.annotationPath + '/' + i)
            for j, a in zip(imgList, annotationList):
                imgPath = self.datasetPath + '/' + i + '/' + j
                annotationPath = self.annotationPath + '/' + i + '/' + a
                # images.append(self.loadImg(imgPath, annotationPath, 224, 224))
                images.append(self.loadImg(imgPath, annotationPath, 299, 299))
                # images.append(self.loadImg(imgPath, annotationPath, 331, 331))
                labels.append(
                    [1 if k == count else 0 for k in range(self.N)])
            count = count + 1
            print("Finish to loading %d category: %s" % (count, i))
        print(len(labels))
        return images, labels

    def loadImg(self, path, annotation=None, x=224, y=224):
        if (annotation != None):
            tree = ET.parse(annotation)
            xmin = int(tree.getroot().findall('.//xmin')[0].text)
            xmax = int(tree.getroot().findall('.//xmax')[0].text)
            ymin = int(tree.getroot().findall('.//ymin')[0].text)
            ymax = int(tree.getroot().findall('.//ymax')[0].text)
            img = Image.open(path)
            img = img.crop((xmin, ymin, xmax, ymax))
            img = img.convert('RGB')
            img = img.resize((x, y), Image.BILINEAR)
        elif (annotation == None):
            img = Image.open(path)
            img = img.convert('RGB')
            img = img.resize((x, y), Image.BILINEAR)
        return img

    def loadTestImg(self, path, x=224, y=224):
        testImg = []
        img = Image.open(path)
        img = img.convert('RGB')
        img = img.resize((x, y), Image.BILINEAR)
        img = np.array(img)
        testImg.append(img)
        testImg = np.array(testImg)
        return testImg

    def showPredictionResult(self, result, percentage, dogs):
        for i in range(len(result)):
            print(str(round(percentage[i]*100, 2)) + '% probability is ' + dogs[result[i]])

    def showReport(self, history):
        # Show the accuracy report.
        plt.figure('Accuracy')
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        # Show the loss report.
        plt.figure('Loss')
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left') 
        plt.show()

        # Show the error report.
        plt.figure('Error')
        plt.plot(history.history['mse'])
        plt.plot(history.history['val_mse'])
        plt.title('Model Mean Squared Error')
        plt.ylabel('mean_squared_error')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left') 
        plt.show()

        # Show the learning rate report.
        plt.figure('Learning Rate')
        plt.plot(history.history['lr'])
        plt.title('Model Learning Rate')
        plt.ylabel('learning rate')
        plt.xlabel('epoch')
        plt.show()