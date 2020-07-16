import json
import numpy as np

class WriteResultServices():
    def __init__(self):
        self.filename = ''

    @staticmethod
    def create(data, filename):
        temp=np.array(data.history).tolist()
        data_json = json.dumps(str(temp))
        f = open("./Result/" + filename + ".json", "w")
        f.write(data_json)
        f.close()