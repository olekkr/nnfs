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
            self.weights.append(0.01 * np.random.randn(self.n_inputs, self.n_neurons))
        self.biases = (np.zeros((self.model.batchsize ,1, self.n_neurons)))
        #print(self.weights, self.biases, "layer params")

    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.output = []
        #print(inputs, "layer input")
        for (inputt, weight, bias) in zip(inputs, self.weights, self.biases):
            self.output.append(np.dot(inputt, weight) + bias)
            #print(inputt.shape, weight.shape. bias)
            #print(inputt.shape, weight.shape, bias.shape)
        print(self.output, "layer out")
        self.fwd.forward(self.output)
        


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
        self.fwd.forward(self.output)
        #print(self.output, "relu out")


# Softmax activation
class Activation_Softmax:
    def init(self):
        pass

    # Forward pass
    def forward(self, inputs):
        self.output = []
        # Get unnormalized probabilities
        for input in inputs:
            exp_values = np.exp(input - np.max(input, axis=1,
                                                keepdims=True))
            # Normalize them for each sample
            probabilities = exp_values / np.sum(exp_values, axis=1,
                                                keepdims=True)
            self.output.append(probabilities)
        
        print(self.output, 'soft out')
        self.fwd.forward(self.output)
    
    


# Cross-entropy loss
class Loss_CategoricalCrossentropy:

    def init(self):
        pass
    # Forward pass
    def forward(self, y_pred):
        self.output = []

        for pred in y_pred:
            # for the sake of testing change later
            y_true = [0] * len(pred[0])
            test = (next(self.model.y_true))[0]
            y_true[test] = 1
            # print(y_true)

            # Number of samples in a batch
            samples = len(pred)

            # Clip data to prevent division by 0
            # Clip both sides to not drag mean towards any value
            y_pred_clipped = np.clip(pred, 1e-7, 1 - 1e-7)

            # Probabilities for target values -
            # only if categorical labels
            # if len(y_true.shape) == 1:
            # correct_confidences = y_pred_clipped[
            # range(samples),
            #    y_true
            # ]

            # Mask values - only for one-hot encoded labels
            # elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

            # Losses
            negative_log_likelihoods = -np.log(correct_confidences)
            self.output.append(negative_log_likelihoods)

        print(self.output, "loss")
