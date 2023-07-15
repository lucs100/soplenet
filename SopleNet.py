import numpy as np
import time

class InputSizeException(Exception):
    """Raised when the size of an input vector does not agree with the size of the input layer"""
    def __init__(self, input, realInput):
        super().__init__(f"Input vector was length {input}, but length {realInput} was expected")

MATRIX_MODE = "C" # C for row-contiguous, F for column-contiguous

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class NeuralNetwork4: #4-layer neural network
    def __init__(self, inputNeuronsCount, hiddenNeurons1Count, hiddenNeurons2Count, outputNeuronsCount):
        self.inputNeuronsCount = inputNeuronsCount
        self.hiddenNeurons1Count = hiddenNeurons1Count
        self.hiddenNeurons2Count = hiddenNeurons2Count
        self.outputNeuronsCount = outputNeuronsCount

        self.h1Weights = np.random.rand(hiddenNeurons1Count, inputNeuronsCount)
        self.h1Biases = np.random.rand(hiddenNeurons1Count, 1)
        self.h2Weights = np.random.rand(hiddenNeurons2Count, hiddenNeurons1Count)
        self.h2Biases = np.random.rand(hiddenNeurons2Count, 1)
        self.outputWeights = np.random.rand(outputNeuronsCount, hiddenNeurons2Count)
        self.outputBiases = np.random.rand(outputNeuronsCount, 1)

        self.inputActivations = np.zeros((inputNeuronsCount, 1), order=MATRIX_MODE)
        self.hidden1Activations = np.zeros((hiddenNeurons1Count, 1), order=MATRIX_MODE)
        self.hidden2Activations = np.zeros((hiddenNeurons2Count, 1), order=MATRIX_MODE)
        self.outputActivations = np.zeros((outputNeuronsCount, 1), order=MATRIX_MODE)
        print("Initialized NeuralNetwork4!")

    def feedforward(self, input):
        """Accepts a vector of input neuron values and returns a vector of output neuron values."""
        if len(input) != self.inputNeuronsCount:
            raise InputSizeException(input, self.inputNeuronsCount) #can remove this later to speed up
        self.hidden1Activations = sigmoid((self.h1Weights @ input) + self.h1Biases)
        self.hidden2Activations = sigmoid((self.h2Weights @ self.hidden1Activations) + self.h2Biases)
        self.outputActivations = sigmoid((self.outputWeights @ self.hidden2Activations) + self.outputBiases)
        return self.outputActivations

start_time = time.time()

testNetwork = NeuralNetwork4(3000, 40, 30, 10)

for i in range(10000):
    (testNetwork.feedforward(np.random.rand(3000, 1)))

print("--- %s seconds ---" % (time.time() - start_time))


