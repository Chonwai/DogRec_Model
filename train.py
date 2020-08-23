import sys
from Services.Utils import Utils
from Services.TrainingMachine import TrainingMachine
import os
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

amount = int(sys.argv[1])
epochs = int(sys.argv[2])
batch = int(sys.argv[3])
gpu = sys.argv[4]

os.environ["CUDA_VISIBLE_DEVICES"] = gpu

def train():
    print(sys.argv)
    utils = Utils(N=amount)
    x, y = utils.loadDataset()
    trainingMachine = TrainingMachine(classN=amount)
    trainingMachine.init(x, y)
    # trainingMachine.trainTFMobileNetV2(epochs, batch)
    # trainingMachine.trainTFVGG19(epochs, batch)
    # trainingMachine.trainTFVGG16(epochs, batch)
    # trainingMachine.trainTFInceptionResNetV2(epochs, batch)
    # trainingMachine.trainTFXception(epochs, batch)
    # trainingMachine.trainTFNASNetLarge(epochs, batch)
    trainingMachine.trainTFNASNetMobile(epochs, batch)
    # trainingMachine.trainNASNetLarge(epochs, batch)
    # trainingMachine.trainMobileNetV2(epochs, batch)

def main():
    train()


if __name__ == '__main__':
    main()
