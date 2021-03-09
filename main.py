import numpy as np
from matplotlib import pyplot
#import mnist 
import pickle

#///https://github.com/alexmeinhold/mnist-parser
#https://github.com/olekkr/nnfs

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
#print(trainData[1][0:100])


class Model:
    def __init__(self, layerList, bs):
        #set input gen and output gen
        self.batchsize = bs

        self.inputs = self.XGen()
        self.y_true = self.YGen()

        self.count = 1 

        self.learningRate = 0.01

        self.layers = layerList

        #set fwd and back
        for l in self.layers:
            l.model = self
            l.init()
            l.isStart = False  
        
        self.layers[0].isStart = True
        
        for idx, l in enumerate(self.layers):
            if idx != len(self.layers)-1:
                l.fwd = self.layers[idx + 1]
            if idx > 0:
                l.bck = self.layers[idx - 1]


    def XGen (self):
        gen = (s for s in trainData[0])
        for _ in gen:
            output = []
            for _ in range(self.batchsize):
                output.append(np.array(next(gen)).ravel())
            yield output
            yield output
            yield output
        

    def YGen (self):
        gen = (s for s in trainData[1])
        for _ in trainData[1]:
            output = []
            for _ in range(self.batchsize):
                y = int(next(gen))
                output.append(y)
            yield output
            yield output
            yield output
    
    def forward(self):
        input = np.array(next(self.inputs))
        self.layers[0].forward(input)
        print(self.layers)

    def backward(self):
        self.layers[-1].backward()


modeltest1 = Model([Layer_Dense(28**2, 10), Activation_ReLU(), Layer_Dense(10, 10), Activation_Softmax(), Loss_CategoricalCrossentropy()], 3) ## , Activation_Softmax()
modeltest1.forward()
modeltest1.backward()
modeltest1.forward()
modeltest1.backward()


#layers = [Layer_Dense(28**2, 64), Activation_ReLU(), Layer_Dense(64, 10),  Activation_ReLU(), Loss_CategoricalCrossentropy()]

#model = Model(layers,1)

#model.forward()

