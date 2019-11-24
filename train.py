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
    transferLearning = TransferLearning(classN=amount)
    transferLearning.init(x, y)
    # transferLearning.trainTFModel(epochs, batch)
    # transferLearning.trainMobileNetV2(epochs, batch)
    # transferLearning.trainXception(epochs, batch)
    transferLearning.trainVGG19(epochs, batch)

def main():
    train()


if __name__ == '__main__':
    main()
