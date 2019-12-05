import sys
from Services.Utils import Utils
from Services.TrainingMachine import TrainingMachine

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

amount = int(sys.argv[1])
path = sys.argv[2]

def test():
    print("Testing Case")
    utils = Utils(N=amount)
    img = utils.loadTestImg(path)
    testing = TrainingMachine(classN=amount)
    testing.eval(img, amount)

def main():
    test()

if __name__ == '__main__':
    main()