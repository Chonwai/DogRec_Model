import os
import matplotlib.pyplot as plt
from PIL import Image
import skimage.io
import skimage.transform
import xml.etree.ElementTree as ET

class Utils():
    def __init__(self, N, datasetPath='./Data/Training', annotationPath='./Data/Annotation'):
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
            for j, k in zip(imgList, annotationList):
                count = count + 1
                # if (count % 2 == 0):
                #     continue
                imgPath = self.datasetPath + '/' + i + '/' + j
                annotationPath = self.annotationPath + '/' + i + '/' + k
                images.append(self.loadImg(imgPath, annotationPath, 168, 168))
                labels.append(
                    [1 if k == count else 0 for k in range(self.N)])
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
        else:
            img = Image.open(path)
            img = img.convert('RGB')
            img = img.resize((x, y), Image.BILINEAR)
        return img

    def loadTestImg(self, path, x=224, y=224):
        img = skimage.io.imread(path)
        img = img / 255.0
        resizeImg = skimage.transform.resize(img, (224, 224, 3))[None, :, :, :]
        return resizeImg
