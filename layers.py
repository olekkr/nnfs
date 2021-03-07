import numpy as np


# Dense layer
class Layer_Dense:

    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases


# ReLU activation
class Activation_ReLU:

    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)


# Softmax activation
class Activation_Softmax:
    # Forward pass
    def forward(self, inputs):
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
                                            keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)
        self.output = probabilities
        print(self.output)


# Cross-entropy loss
class Loss_CategoricalCrossentropy:

    # Forward pass
    def forward(self, y_pred):
        

        ## for the sake of testing change later
        y_true = [0] * len(y_pred[0])
        y_true[next(self.model.y_true)[0]] = 1

        
        # Number of samples in a batch
        samples = len(y_pred)

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

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
            y_pred_clipped*y_true,
            axis=1
        )

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        self.output = negative_log_likelihoods

        print(self.output)
