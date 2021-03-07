import numpy as np
from matplotlib import pyplot
import mnist
import pickle

from layers import *


# ##### Caching databse into a pickle
#with open("pickle", 'wb') as f:
#    (trainData, testData) = mnist.load_data()
#    pickle.dump([trainData[0][:1000], trainData[1][:1000]], f)
    
#Use cached data
with open("./pickle", 'rb') as f:
    trainData = pickle.load(f)


#Without using cached data (slow)
#(x_train, y_train) = pickle.load(f)

#pyplot.imshow(trainData[0][9])
#pyplot.show()


class Model:
    def __init__(self, layerList, bs):
        #set input gen and output gen
        self.batchsize = bs

        self.inputs = self.XGen()
        self.y_true = self.YGen()

        self.count = 1 

        self.layers = layerList
        #set fwd and back
        for l in self.layers:
            l.model = self
        
        for idx, l in enumerate(self.layers):
            if idx != len(layers)-1:
                l.fwd = layers[idx + 1]
            if idx != 0:
                l.bck = layers[idx - 1]
    
    def XGen (self):
        gen = (s for s in trainData[0])
        for i in gen:
            output = []
            for ii in range(self.batchsize):
                output.append(np.array(next(gen)).ravel())
            yield output

    def YGen (self):
        gen = (s for s in trainData[1])
        for i in gen:
            output = []
            for ii in range(self.batchsize):
                output.append(int(next(gen)))
            yield output
    
    def forward(self):
        input = next(self.inputs)
        self.layers[0].forward(np.array(input))
        print(self.layers)
        #last = self.layers[0]
        #for item in self.layers[1:]:
        #    item.forward(last.output)
        #    last = item


layers = [Layer_Dense(28**2, 64), Activation_ReLU(), Layer_Dense(64, 10),  Activation_Softmax(), Loss_CategoricalCrossentropy()]

model = Model(layers,1)

model.forward()

