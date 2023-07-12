import numpy as np

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

        self.inputActivations = np.zeros((inputNeuronsCount, 1), order='F')
        self.hidden1Activations = np.zeros((hiddenNeurons1Count, 1), order='F')
        self.hidden2Activations = np.zeros((hiddenNeurons2Count, 1), order='F')
        self.outputActivations = np.zeros((outputNeuronsCount, 1), order='F')
        print("Initialized NeuralNetwork4!")

testNetwork = NeuralNetwork4(2, 3, 4, 5)
pass