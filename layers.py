import numpy as np


# Dense layer
class Layer_Dense:

    # Layer initialization
    def __init__(self, ninputs, nneurons):
        (self.n_inputs, self.n_neurons) = ninputs, nneurons

    # Initialize weights and biases

    def init(self):
        self.weights = []
        for i in range(self.model.batchsize):
            self.weights.append(
                0.01 * np.random.randn(self.n_inputs, self.n_neurons))
        self.biases = (np.zeros((self.model.batchsize, 1, self.n_neurons)))

    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.inputs = inputs
        self.output = []
        for (inputt, weight, bias) in zip(inputs, self.weights, self.biases):
            self.output.append(np.dot(inputt, weight) + bias)
        self.fwd.forward(self.output)
        #print(np.array(self.output).shape)

    def backward(self):
        #print(self)
        
        #print('weights:', np.array(self.weights).shape, 'inputs', np.array(self.inputs).shape, self.biases, "params")

        #print(np.array(self.fwd.derivatives).shape, "testtestestder")


        #dL/dw 
        #for each elem in batch
        dLdw = []
        for db , ib in zip(self.fwd.derivatives, self.inputs):
            dLdw.append(np.dot(np.array(db).T, np.array(ib)))
            #new in batch
            #new = []
            #for i in ib:
            #    ds = np.multiply(ds, i)
            #    new.append(i*np.array(db[0]))
            #dLdw.append(new)
        print(np.array(dLdw).shape,dLdw)

        #dL/db
        dLdb = self.fwd.derivatives 
        print(dLdb)

        
        #dL/da is w * dL/dz
        #self.derivatives = self.weights * self.fwd.derivatives
        
        #self.weights -= np.array([np.average(np.array(dLdw) * self.model.learningRate, axis=0)] * self.model.batchsize)
        print()
        self.biases -= np.array([np.average(np.array(dLdb) * self.model.learningRate, axis=0)] * self.model.batchsize)

        #print(dLdb, self.biases.shape)

        if(not self.isStart):
            self.bck.backward()




# ReLU activation
class Activation_ReLU:

    def __init__(self):
        pass

    def init(self):
        pass
    # Forward pass

    def forward(self, inputs):
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)
        #print(self.output, "relu out")
        self.fwd.forward(self.output)
    
    def backward(self):
        #print(self)
        self.derivatives = np.clip(self.output, 0, 1)
        self.bck.backward()
        pass

# Softmax activation
class Activation_Softmax:
    def init(self):
        pass

    # Forward pass
    def forward(self, inputs):
        self.inputs = inputs
        self.output = []
        # Get unnormalized probabilities
        #print(inputs)
        for input in inputs:
            exp_values = np.exp(input - np.max(input, axis=1,
                                               keepdims=True))
            # Normalize them for each sample
            probabilities = exp_values / np.sum(exp_values, axis=1,
                                                keepdims=True)
            self.output.append(probabilities)

        #print(self.output, 'soft out')
        self.fwd.forward(self.output)

    def backward(self):
        #print(self)
        # do for each elem in batch
        # da(x)/d(notx)
        # will be calculate dL/dz for each
        self.derivatives = self.output
        for (d, y) in zip(self.derivatives, self.fwd.y_trues):
            #print(d,y, "howdy")
            d[0][y] = d[0][y] - d[0][y]**2       
        self.derivatives = np.multiply(self.derivatives, self.fwd.derivatives) 
        #print(self.derivatives, self.fwd.y_trues, "der, trues")
        self.bck.backward()


class Loss_CategoricalCrossentropy:

    def init(self):
        pass
    # Forward pass

    def forward(self, y_pred):
        self.y_pred = y_pred
        self.y_trues = next(self.model.y_true)
        self.trueVects = []
        self.output = []

        nOutputs = len(y_pred[0][0])

        for pred, true in zip(self.y_pred, self.y_trues):
            # create onehot vector
            trueVect = [0] * nOutputs
            trueVect[true] = 1

            # appends truevect for future use
            self.trueVects.append(np.clip((trueVect), 1e-7, 1 - 1e-7))

            # Number of samples in a batch

            # Clip data to prevent division by 0
            # Clip both sides to not drag mean towards any value
            y_pred_clipped = np.clip(pred, 1e-7, 1 - 1e-7)

            # Mask values - only for one-hot encoded labels
            # elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * trueVect,
                axis=1
            )

            # Losses
            negative_log_likelihoods = -np.log(correct_confidences)
            self.output.append(negative_log_likelihoods)
        print(self.output, "loss")
        print([np.argmax(a) for a in y_pred], self.y_trues)

    def backward(self):
        #print(self)
        # da(x)/dn
        # will be calculate dL/dz for each
        self.derivatives = []
        print('test', self.y_pred[0], self.trueVects)
        for (p, y) in zip(self.y_pred, self.trueVects):
            self.derivatives.append(-y/np.clip(p, 1e-7, 1 - 1e-7))
            #print(self.derivatives, "der")
        self.bck.backward()
