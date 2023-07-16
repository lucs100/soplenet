import numpy as np
import time

np.set_printoptions(edgeitems=30, linewidth=100000)
rng = np.random.default_rng()

class InputSizeException(Exception):
    """Raised when the size of an input vector does not agree with the size of the input layer"""
    def __init__(self, input, realInput):
        super().__init__(f"Input vector was length {input}, but length {realInput} was expected")

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

        self.h1Weights = rng.random((h1NeuronCount, inputNeuronCount))*2-1
        self.h1Biases = rng.random((h1NeuronCount,))*2-1

        self.h2Weights = rng.random((h2NeuronCount, h1NeuronCount))*2-1
        self.h2Biases = rng.random((h2NeuronCount,))*2-1

        self.outputWeights = rng.random((outputNeuronCount, h2NeuronCount))*2-1
        self.outputBiases = rng.random((outputNeuronCount,))*2-1

        #Keep initialized to save time
        self.inputActivations = np.zeros((inputNeuronCount,))
        self.h1Activations = np.zeros((h1NeuronCount,))
        self.h2Activations = np.zeros((h2NeuronCount,))
        self.outputActivations = np.zeros((outputNeuronCount,))

        self.outputTarget = np.zeros((outputNeuronCount,))

        self.h1Error = np.zeros((inputNeuronCount,))
        self.h2Error = np.zeros((inputNeuronCount,))
        self.outputError = np.zeros((inputNeuronCount,))

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
        """Returns the scalar cost of the last network feedforward."""
        return np.sum((self.outputActivations-self.outputTarget)**2)

    def backpropagateError(self):
        """Computes each layer's error vector via backpropagation."""
        self.outputError = 2*(self.outputActivations-self.outputTarget)*dSigmoid(self.outputRaw)

        self.h2Error = (self.outputWeights.transpose() @ self.outputError) * dSigmoid(self.h2Raw)
        self.h1Error = (self.h2Weights.transpose() @ self.h2Error) * dSigmoid(self.h1Raw)
    
    def getBiasGradient(self):
        """Calculates and returns 3 gradient-like vectors for the biases in the network."""
        biasGradientH1 = self.h1Error * self.h1Biases
        biasGradientH2 = self.h2Error * self.h2Biases
        biasGradientOutput = self.outputError * self.outputBiases
        return [biasGradientH1, biasGradientH2, biasGradientOutput]
    
    def getWeightGradient(self):
        """Calculates and returns 3 gradient-like matrices for the weights in the network."""
        weightGradientH1 = self.inputActivations * self.h1Error[:, np.newaxis]  #Equivalent of transpose, creates a meshgrid-like sum.
        weightGradientH2 = self.h1Activations * self.h2Error[:, np.newaxis]
        weightGradientOutput = self.h2Activations * self.outputError[:, np.newaxis]
        return [weightGradientH1, weightGradientH2, weightGradientOutput]

    def adjustBiases(self, gradient, eta):
        """Adjusts the biases given some gradient-like vectors of the biases and some training rate eta. Gradient is a list of 3 vectors."""
        biasGradientH1, biasGradientH2, biasGradientOutput = gradient
        self.h1Biases -= (biasGradientH1)*eta
        self.h2Biases -= (biasGradientH2)*eta
        self.outputBiases -= (biasGradientOutput)*eta

    def adjustWeights(self, gradient, eta):
        """Adjusts the weights given some gradient-like matrices of the weights and some training rate eta. Gradient is a list of 3 matrices."""
        weightGradientH1, weightGradientH2, weightGradientOutput = gradient
        self.h1Weights -= (weightGradientH1)*eta
        self.h2Weights -= (weightGradientH2)*eta
        self.outputWeights -= (weightGradientOutput)*eta
    


#start_time = time.time()
testNetwork = NeuralNetwork4(5, 6, 7, 8)
testNetwork.outputTarget = np.array([0, 1, 0, 0, 0, 0, 0, 0])
# print(f"Output EVec: {testNetwork.outputError}")
# print(f"Hidden2 EVec: {testNetwork.h2Error}")
# print(f"Hidden1 EVec: {testNetwork.h1Error}")

#Example training:
for i in range(1000):
    testNetwork.feedforward(np.linspace(0.1, 0.9, 5))
    testNetwork.backpropagateError()

    biasGradient = testNetwork.getBiasGradient()
    weightGradient = testNetwork.getWeightGradient()

    testNetwork.adjustBiases(biasGradient, eta=0.01)
    testNetwork.adjustWeights(weightGradient, eta=0.01)

    print(f"Current cost: {testNetwork.getIterationCost()} \t {testNetwork.outputActivations}")

# print("--- %s seconds ---" % (time.time() - start_time))


