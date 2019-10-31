import os
from Services.Utils import Utils
from Services.TransferLearning import TransferLearning

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

amount = 120

def main():
    utils = Utils(N=amount)
    x, y = utils.loadDataset()
    transferLearning = TransferLearning(x, y, classN=amount)
    transferLearning.init()
    transferLearning.trainVGG()

if __name__ == '__main__':
    main()
