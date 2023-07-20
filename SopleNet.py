import numpy as np
import time
from dataclasses import dataclass
import pickle
import csv
import os
from datetime import datetime

np.set_printoptions(edgeitems=30, linewidth=100000)

TIMESTAMP = datetime.now().strftime("%Y%b%d_%Hh%Mm%Ss")
os.makedirs(f"./results/{TIMESTAMP}/")

RNG_MEAN = 0
RNG_STDDEV = 1
rng = np.random.default_rng()

CIFAR_DATA_TRAIN: np.array
CIFAR_DATA_TEST: np.array
CIFAR_LABELS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

INPUT_LENGTH = 3072
OUTPUT_LENGTH = 10

@dataclass
class TestImage:
    data: np.array
    label: int

    def scaledData(self):
        return (self.data / 255)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict    

CIFAR_DATA_TRAIN = np.array(unpickle("cifar10TRAIN"))
CIFAR_DATA_TEST = np.array(unpickle("cifar10TEST"))

CIFAR_DATA_TRAIN_SAMPLES = len(CIFAR_DATA_TRAIN)
CIFAR_DATA_TEST_SAMPLES = len(CIFAR_DATA_TEST)

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

        self.h1Weights = rng.normal(RNG_MEAN, RNG_STDDEV, (h1NeuronCount, inputNeuronCount))
        self.h1Biases = rng.normal(RNG_MEAN, RNG_STDDEV, (h1NeuronCount,))

        self.h2Weights = rng.normal(RNG_MEAN, RNG_STDDEV, (h2NeuronCount, h1NeuronCount))
        self.h2Biases = rng.normal(RNG_MEAN, RNG_STDDEV, (h2NeuronCount,))

        self.outputWeights = rng.normal(RNG_MEAN, RNG_STDDEV, (outputNeuronCount, h2NeuronCount))
        self.outputBiases = rng.normal(RNG_MEAN, RNG_STDDEV, (outputNeuronCount,))

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

    def feedforward(self, input) -> np.array:
        """Accepts a vector of input neuron values and returns a vector of output neuron values."""
        if len(input) != self.inputNeuronCount:
            raise InputSizeException(len(input), self.inputNeuronCount) #can remove this later to speed up
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
    
    def getIterationCost(self) -> None:
        """Returns the scalar cost of the last network feedforward."""
        return np.sum((self.outputActivations-self.outputTarget)**2)

    def backpropagateError(self) -> None:
        """Computes each layer's error vector via backpropagation."""
        self.outputError = -(self.outputTarget-self.outputActivations) #output layer error is special

        self.h2Error = (self.outputWeights.transpose() @ self.outputError) * dSigmoid(self.h2Raw) #each L is based on L+1
        self.h1Error = (self.h2Weights.transpose() @ self.h2Error) * dSigmoid(self.h1Raw)
    
    def getBiasGradient(self) -> list[np.array]:
        """Calculates and returns 3 gradient-like vectors for the biases in the network."""
        biasGradientH1 = self.h1Error * self.h1Biases
        biasGradientH2 = self.h2Error * self.h2Biases
        biasGradientOutput = self.outputError * self.outputBiases
        return [biasGradientH1, biasGradientH2, biasGradientOutput]
    
    def getWeightGradient(self) -> list[np.array]:
        """Calculates and returns 3 gradient-like matrices for the weights in the network."""
        weightGradientH1 = self.inputActivations * self.h1Error[:, np.newaxis]  #Equivalent of transpose, creates a meshgrid-like sum.
        weightGradientH2 = self.h1Activations * self.h2Error[:, np.newaxis]
        weightGradientOutput = self.h2Activations * self.outputError[:, np.newaxis]
        return [weightGradientH1, weightGradientH2, weightGradientOutput]

    def adjustBiases(self, gradient, eta) -> None:
        """Adjusts the biases given some gradient-like vectors of the biases and some training rate eta. Gradient is a list of 3 vectors."""
        biasGradientH1, biasGradientH2, biasGradientOutput = gradient
        self.h1Biases -= (biasGradientH1)*eta
        self.h2Biases -= (biasGradientH2)*eta
        self.outputBiases -= (biasGradientOutput)*eta

    def adjustWeights(self, gradient, eta) -> None:
        """Adjusts the weights given some gradient-like matrices of the weights and some training rate eta. Gradient is a list of 3 matrices."""
        weightGradientH1, weightGradientH2, weightGradientOutput = gradient
        self.h1Weights -= (weightGradientH1)*eta
        self.h2Weights -= (weightGradientH2)*eta
        self.outputWeights -= (weightGradientOutput)*eta
    
class NeuralNetwork4Trainer:
    def __init__(self, network: NeuralNetwork4, defaultTrainingRate: float):
        self.network = network
        self.globalCostHistory = []
        self.samples = 0
        self.correct = 0
        self.defaultTrainingRate = defaultTrainingRate

        print("Initialized NeuralNetwork4Trainer!")

    def setupTrainingData(self, data: list[TestImage]) -> None:
        """Accepts a list of data, and stores it in the trainer."""
        self.trainingData = data
        self.trainingDataSize = len(data)

    def __trainMiniBatch(self, miniBatch: list[TestImage], trainingRate: int) -> None:
        """Begins training the network using loaded training data."""
        #print(f"Beginning training on minibatch {self.miniBatchIdx}!")

        caseIdx = 0
        correct = 0
        localCostHistory = []
            
        weightGradientH1History = []
        weightGradientH2History = []
        weightGradientOutputHistory = []
        biasGradientH1History = []
        biasGradientH2History = []
        biasGradientOutputHistory = []

        testCase: TestImage
        for testCase in miniBatch:
            self.network.outputTarget = np.zeros_like(self.network.outputTarget)
            self.network.outputTarget[testCase.label] = 1

            decision = np.argmax(self.network.feedforward(testCase.scaledData()))
            self.network.backpropagateError()
            
            real = (decision == testCase.label)
            if real:
                correct += 1
                self.correct += 1
                self.epochCorrect += 1

            localCostHistory.append(self.network.getIterationCost())

            biasGradient = self.network.getBiasGradient()
            weightGradient = self.network.getWeightGradient()

            weightGradientH1History.append(weightGradient[0])
            weightGradientH2History.append(weightGradient[1])
            weightGradientOutputHistory.append(weightGradient[2])
            biasGradientH1History.append(biasGradient[0])
            biasGradientH2History.append(biasGradient[1])
            biasGradientOutputHistory.append(biasGradient[2])

            if (caseIdx == 1 and SUPER_VERBOSE):
                print(f"Training... MB: {self.miniBatchIdx} \t Obj: {caseIdx} \t Cost: {self.network.getIterationCost()} \t Result: {real} [{decision}R, {testCase.label}E]")
            
            caseIdx += 1
            self.samples += 1
            self.epochSamples += 1

        acc = correct/(len(miniBatch))
        avgCost = np.mean(localCostHistory)
        self.globalCostHistory.append(avgCost)
        self.epochCostSum += avgCost

        wH1d = np.mean(weightGradientH1History, axis=0)
        wH2d = np.mean(weightGradientH2History, axis=0)
        wOd = np.mean(weightGradientOutputHistory, axis=0)
        bH1d = np.mean(biasGradientH1History, axis=0)
        bH2d = np.mean(biasGradientH2History, axis=0)
        bOd = np.mean(biasGradientOutputHistory, axis=0)

        self.network.adjustWeights([wH1d, wH2d, wOd], eta=trainingRate)
        self.network.adjustBiases([bH1d, bH2d, bOd], eta=trainingRate)

        if VERBOSE:
            print(f"Minibatch {self.miniBatchIdx}/{self.miniBatchCount} complete! \t Cost: {avgCost} \t Acc: {int(acc*100)}% \t Elapsed: {round(time.time() - start_time, 3)}s")
        
    def generateMiniBatches(self, miniBatchSize):
        rng.shuffle(self.trainingData)
        self.miniBatchSet = np.array_split(self.trainingData, (self.trainingDataSize // miniBatchSize))
        self.miniBatchSize = len(self.miniBatchSet[0])
        self.miniBatchCount = len(self.miniBatchSet)
        self.miniBatchIdx = 0

    def beginTraining(self, epochCount: int, miniBatchSize: int) -> None:
        """Begins training the network with each loaded minibatch."""

        self.currentEpoch = 1
        while self.currentEpoch <= epochCount:
            if (not VERBOSE and not SUPER_VERBOSE):
                print(f"Beginning Epoch {self.currentEpoch}/{epochCount}... ", end="")
            self.generateMiniBatches(miniBatchSize)
            self.epochSamples = 0
            self.epochCorrect = 0
            self.epochCostSum = 0
            for miniBatch in self.miniBatchSet:
                self.miniBatchIdx += 1
                self.__trainMiniBatch(miniBatch, self.defaultTrainingRate)
            print(f"done. \t Average accuracy:{round(100*self.epochCorrect/self.epochSamples, 3)}% \t Elapsed: {round(time.time() - start_time, 3)}s")
            epochCost = self.epochCostSum / self.miniBatchCount
            logEpoch(self.currentEpoch, epochCount, round(epochCost, 4), round(100*self.epochCorrect/self.epochSamples, 3), round(time.time() - start_time, 3))
            self.currentEpoch += 1
        print(f"Training complete! \t Average accuracy:{round(100*self.correct/self.samples, 3)}%")
        self.tMax = round(time.time() - start_time, 3)
    
    def __testNetwork(self, testSet: list[TestImage]):
        caseIdx = 0
        correct = 0
        testSamples = len(testSet)
        correctSet = np.zeros(10,)
        sampleSet = np.zeros(10,)
        predSet = np.zeros(10,)

        print(f"\n\n\nTesting in progress...")

        for item in testSet:

            prediction = np.argmax(self.network.feedforward(item.scaledData()))
            if (prediction == item.label):
                correct += 1
                correctSet[item.label] += 1

            caseIdx += 1
            sampleSet[item.label] += 1
            predSet[prediction] += 1
        
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        print("\nTesting complete!")
        print(f"Per-category accuracy: \t \t {correctSet*100 / sampleSet}")
        print(f"Per-category predictions: \t {predSet*100 / testSamples}")
        print(f"Total accuracy: \t \t {correct} samples correct out of {testSamples} samples [{correct*100/testSamples}%]")
        with open(f"./results/{TIMESTAMP}/TEST_LOG.csv", "w", newline="\n") as file:
            file.write("Testing complete! \n")
            file.write(f"Per-category accuracy: \t \t {correctSet*100 / sampleSet} \n")
            file.write(f"Per-category predictions: \t {predSet*100 / testSamples} \n")
            file.write(f"Total accuracy: \t \t {correct} samples correct out of {testSamples} samples [{correct*100/testSamples}%] \n")
    
    def beginTesting(self, data: list[TestImage]) -> None:
        """Begins testing the network against the specified testset. 
        Only accuracy is reported, no backpropagation is performed.
        Does not affect the network; is idempotent."""
        self.__testNetwork(data)

def setupEpochLogging():
    with open(f"./results/{TIMESTAMP}/trainingLog.csv", "a+", newline="\n") as csvfile:
        sopleWriter = csv.writer(csvfile, delimiter = ' ')
        sopleWriter.writerow(["Epoch", "MaxEpoch", "AvgCost", "EpochAcc", "Elapsed"])
    csvfile.close()

def logEpoch(epoch, maxEpoch, avgCost, epochAcc, elapsed):
    with open(f"./results/{TIMESTAMP}/trainingLog.csv", "a+", newline="\n") as csvfile:
        sopleWriter = csv.writer(csvfile, delimiter = ' ')
        sopleWriter.writerow([epoch, maxEpoch, avgCost, epochAcc, elapsed])
    csvfile.close()

VERBOSE = True
SUPER_VERBOSE = False

layerH1NeuronCount = 2
layerH2NeuronCount = 2
trainingRate = 0.005
miniBatchSize = 100
epochCount = 3

setupEpochLogging()

cifarNetwork = NeuralNetwork4(INPUT_LENGTH, layerH1NeuronCount, layerH2NeuronCount, OUTPUT_LENGTH)
cifarNetworkTrainer = NeuralNetwork4Trainer(cifarNetwork, trainingRate)
cifarNetworkTrainer.setupTrainingData(CIFAR_DATA_TRAIN)

start_time = time.time()
cifarNetworkTrainer.beginTraining(epochCount, miniBatchSize)
cifarNetworkTrainer.beginTesting(CIFAR_DATA_TEST)

with open(f"./results/{TIMESTAMP}/cifarModelTrained", "wb") as file:
    pickle.dump(cifarNetworkTrainer, file=file)
with open(f"./results/{TIMESTAMP}/hyperparameters", "wb") as file:
    pickle.dump([RNG_MEAN, RNG_STDDEV, layerH1NeuronCount, layerH2NeuronCount, trainingRate, miniBatchSize, epochCount], file=file)

