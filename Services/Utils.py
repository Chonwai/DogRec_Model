import os
import skimage.io
import skimage.transform


class Utils():
    def __init__(self, N, datasetPath='./Data/Training'):
        self.datasetPath = datasetPath
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
            for j in imgList:
                if (count % 2 == 0):
                    continue
                path = self.datasetPath + '/' + i + '/' + j
                images.append(self.loadImg(path, 56, 56))
                labels.append(
                    [1 if k == count else 0 for k in range(self.N)])
            print("Finish to loading %d category: %s" % (count, i))
            count = count + 1
        print(len(labels))
        return images, labels

    def loadImg(self, path, x=224, y=224):
        img = skimage.io.imread(path)
        img = img / 255.0
        resizeImg = skimage.transform.resize(img, (x, y, 3))
        return resizeImg

    def loadTestImg(self, path, x=224, y=224):
        img = skimage.io.imread(path)
        img = img / 255.0
        resizeImg = skimage.transform.resize(img, (224, 224, 3))[None, :, :, :]
        return resizeImg
