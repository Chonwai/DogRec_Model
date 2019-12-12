import sys
from Services.Utils import Utils
from Services.TrainingMachine import TrainingMachine

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

amount = int(sys.argv[1])
epochs = int(sys.argv[2])
batch = int(sys.argv[3])

def train():
    print(sys.argv)
    utils = Utils(N=amount)
    x, y = utils.loadDataset()
    trainingMachine = TrainingMachine(classN=amount)
    trainingMachine.init(x, y)
    # trainingMachine.trainTFVGG19(epochs, batch)
    trainingMachine.trainTFMobileNetV2(epochs, batch)
    # trainingMachine.trainMobileNetV2(epochs, batch)
    # trainingMachine.trainXception(epochs, batch)
    # trainingMachine.trainVGG19(epochs, batch)

def main():
    train()


if __name__ == '__main__':
    main()
