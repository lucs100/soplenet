import numpy as np
import time

np.set_printoptions(edgeitems=30, linewidth=100000)

class InputSizeException(Exception):
    """Raised when the size of an input vector does not agree with the size of the input layer"""
    def __init__(self, input, realInput):
        super().__init__(f"Input vector was length {input}, but length {realInput} was expected")

MATRIX_MODE = "C" # C for row-contiguous, F for column-contiguous

def sigmoid(x):
    """The sigmoid function, a 'squishification' function for activations."""
    return 1 / (1 + np.exp(-x))

def dSigmoid(x):
    """The derivative of the sigmoid function."""
    return sigmoid(x) * (1-sigmoid(x))

class NeuralNetwork4: #4-layer neural network
    def __init__(self, inputNeuronCount, h1NeuronCount, h2NeuronCount, outputNeuronCount):
        self.inputNeuronCount = inputNeuronCount
        self.h1NeuronCount = h1NeuronCount
        self.h2NeuronCount = h2NeuronCount
        self.outputNeuronCount = outputNeuronCount

        self.h1Weights = (np.random.rand(h1NeuronCount, inputNeuronCount)*2)-1
        self.h1Biases = (np.random.rand(h1NeuronCount, 1)*2)-1
        self.h2Weights = (np.random.rand(h2NeuronCount, h1NeuronCount)*2)-1
        self.h2Biases = (np.random.rand(h2NeuronCount, 1)*2)-1
        self.outputWeights = (np.random.rand(outputNeuronCount, h2NeuronCount)*2)-1
        self.outputBiases = (np.random.rand(outputNeuronCount, 1)*2)-1

        #Keep initialized to save time
        self.inputActivations = np.zeros((inputNeuronCount, 1), order=MATRIX_MODE)
        self.h1Activations = np.zeros((h1NeuronCount, 1), order=MATRIX_MODE)
        self.h2Activations = np.zeros((h2NeuronCount, 1), order=MATRIX_MODE)
        self.outputActivations = np.zeros((outputNeuronCount, 1), order=MATRIX_MODE)

        self.outputTarget = np.zeros((outputNeuronCount, 1), order=MATRIX_MODE)

        self.h1Error = np.zeros((inputNeuronCount, 1), order=MATRIX_MODE)
        self.h2Error = np.zeros((inputNeuronCount, 1), order=MATRIX_MODE)
        self.outputError = np.zeros((inputNeuronCount, 1), order=MATRIX_MODE)

        #Keep for backprop
        self.h1Raw = self.h1Activations
        self.h2Raw = self.h2Activations
        self.outputRaw = self.outputActivations
        print("Initialized NeuralNetwork4!")

    def feedforward(self, input):
        """Accepts a vector of input neuron values and returns a vector of output neuron values."""
        if len(input) != self.inputNeuronCount:
            raise InputSizeException(input, self.inputNeuronCount) #can remove this later to speed up
        self.inputActivations = input

        #Calculate the first layer
        self.h1Raw = (self.h1Weights @ self.inputActivations) + self.h1Biases
        self.h1Activations = sigmoid(self.h1Raw)

        #Calculate the second layer
        self.h2Raw = (self.h2Weights @ self.h1Activations) + self.h2Biases
        self.h2Activations = sigmoid(self.h2Raw)

        #Calculate the output layer
        self.outputRaw = (self.outputWeights @ self.h2Activations) + self.outputBiases
        self.outputActivations = sigmoid(self.outputRaw)

        return self.outputActivations
    
    def getIterationCost(self):
        return np.sum((self.outputActivations-self.outputTarget)**2)

    def backpropagateError(self):
        """Computes each layer's error vector via backpropagation."""
        self.outputError = np.concatenate(2*(self.outputActivations-self.outputTarget)*dSigmoid(self.outputRaw)).transpose()

        self.h2Error = np.concatenate((self.outputWeights.transpose() @ self.outputError).reshape(self.h2NeuronCount, 1) * dSigmoid(self.h2Raw).reshape(self.h2NeuronCount, 1))
        self.h1Error = np.concatenate((self.h2Weights.transpose() @ self.h2Error).reshape(self.h1NeuronCount, 1) * dSigmoid(self.h1Raw).reshape(self.h1NeuronCount, 1))

    

#start_time = time.time()
testNetwork = NeuralNetwork4(5, 6, 7, 8)
testNetwork.feedforward(np.random.rand(5, 1))
testNetwork.outputTarget = np.array([0, 1, 0, 0, 0, 0, 0, 0], ndmin=2).transpose()
testNetwork.backpropagateError()
print(f"Output EV: {testNetwork.outputError}")
print(f"Hidden2 EV: {testNetwork.h2Error}")
print(f"Hidden1 EV: {testNetwork.h1Error}")

print(testNetwork.getIterationCost())
# for i in range(10000):
#     (testNetwork.feedforward(np.random.rand(3000, 1)))

# print("--- %s seconds ---" % (time.time() - start_time))


