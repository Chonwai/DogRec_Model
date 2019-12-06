import os
import matplotlib.pyplot as plt
from PIL import Image
import skimage.io
import skimage.transform
import xml.etree.ElementTree as ET

class Utils():
    def __init__(self, N=120, datasetPath='./Data/Training', annotationPath='./Data/Annotation'):
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

    def loadDataset(self):
        images = []
        labels = []
        self.readDatasetFolderName()
        count = 0
        for i in self.folderList[:self.N]:
            imgList = os.listdir(self.datasetPath + '/' + i)
            annotationList = os.listdir(self.annotationPath + '/' + i)
            for j, a in zip(imgList, annotationList):
                imgPath = self.datasetPath + '/' + i + '/' + j
                annotationPath = self.annotationPath + '/' + i + '/' + a
                images.append(self.loadImg(imgPath, annotationPath, 160, 160))
                # images.append(self.loadImg(path=imgPath, x=160, y=160))
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
        # img = skimage.io.imread(path)
        img = Image.open(path)
        img = img.convert('RGB')
        img = img.resize((x, y), Image.BILINEAR)
        # resizeImg = skimage.transform.resize(img, (224, 224, 3))[None, :, :, :]
        return img

    def showReport(self, history):
        # Show the accuracy report.
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left') 
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left') 
        plt.show()

        plt.plot(history.history['mean_squared_error'])
        plt.plot(history.history['val_mean_squared_error'])
        plt.title('Model Mean Squared Error')
        plt.ylabel('mean_squared_error')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left') 
        plt.show()

        plt.plot(history.history['lr'])
        plt.title('Model Learning Rate')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()







