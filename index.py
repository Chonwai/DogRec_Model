import sys
from Services.Utils import Utils
from Services.TransferLearning import TransferLearning

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

amount = int(sys.argv[1])
epochs = int(sys.argv[2])
batch = int(sys.argv[3])

def train():
    print(sys.argv)
    utils = Utils(N=amount)
    x, y = utils.loadDataset()
    transferLearning = TransferLearning(x, y, classN=amount)
    transferLearning.init()
    transferLearning.trainTFModel(epochs, batch)

def test():
    print("Testing Case")
    utils = Utils(N=amount)
    img = utils.loadImg('./Test/testImg01.jpg')
    testing = TransferLearning(classN=amount)
    testing.eval(img, 5)

def main():
    test()


if __name__ == '__main__':
    main()
