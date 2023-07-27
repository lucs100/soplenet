import os
import numpy as np
import time
from dataclasses import dataclass
import pickle
import SopleNetLogging as logging

np.set_printoptions(edgeitems=30, linewidth=100000)

START_TIME = time.time()

rng = np.random.default_rng()

EPSILON = 0.000000000000001

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

def unpickle(filepath):
    with open(filepath, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    fo.close()
    return dict

def repickle(filepath, data):
    with open(filepath, 'wb') as file:
        pickle.dump(data, file=file)
    file.close()

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
        return -np.sum(self.outputTarget * np.log(self.outputActivations + EPSILON) 
            + (1-self.outputTarget)*(np.log((1-self.outputActivations)+EPSILON)))

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
    
    def logParameters(self) -> 'NeuralNetwork4File': #string name solves circular dependency
        return NeuralNetwork4File(self)
    
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
        batchCorrect = 0
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
                batchCorrect += 1

            caseCost = self.network.getIterationCost()

            localCostHistory.append(caseCost)

            biasGradient = self.network.getBiasGradient()
            weightGradient = self.network.getWeightGradient()

            weightGradientH1History.append(weightGradient[0])
            weightGradientH2History.append(weightGradient[1])
            weightGradientOutputHistory.append(weightGradient[2])
            biasGradientH1History.append(biasGradient[0])
            biasGradientH2History.append(biasGradient[1])
            biasGradientOutputHistory.append(biasGradient[2])

            if (LOGGING_LEVEL > 2):
                self.logPointComplete(caseIdx, caseCost, real, decision, testCase.label)
                
            caseIdx += 1
            self.samples += 1

        acc = int(batchCorrect/(len(miniBatch))*100)
        avgCost = np.mean(localCostHistory)
        self.globalCostHistory.append(avgCost)
        self.epochCostSum += avgCost
        self.epochCorrect += batchCorrect
        self.correct += batchCorrect #TODO: please refactor this.. only used in log training

        wH1d = np.mean(weightGradientH1History, axis=0)
        wH2d = np.mean(weightGradientH2History, axis=0)
        wOd = np.mean(weightGradientOutputHistory, axis=0)
        bH1d = np.mean(biasGradientH1History, axis=0)
        bH2d = np.mean(biasGradientH2History, axis=0)
        bOd = np.mean(biasGradientOutputHistory, axis=0)

        self.network.adjustWeights([wH1d, wH2d, wOd], eta=trainingRate)
        self.network.adjustBiases([bH1d, bH2d, bOd], eta=trainingRate)

        if LOGGING_LEVEL > 1 :
            self.logMinibatchComplete(avgCost, acc)

    def generateMiniBatches(self, miniBatchSize):
        rng.shuffle(self.trainingData)
        self.miniBatchSet = np.array_split(self.trainingData, (self.trainingDataSize // miniBatchSize))
        self.miniBatchSize = len(self.miniBatchSet[0])
        self.miniBatchCount = len(self.miniBatchSet)
        self.miniBatchIdx = 0  

    def beginTraining(self, epochCount: int, miniBatchSize: int) -> None:
        """Begins training the network with each loaded minibatch."""
        print("Beginning training...")
        self.epochCount = epochCount
        for self.currentEpoch in range(1, epochCount+1):
            self.generateMiniBatches(miniBatchSize)
            self.epochCorrect = 0
            self.epochCostSum = 0
            for miniBatch in self.miniBatchSet:
                self.miniBatchIdx += 1
                self.__trainMiniBatch(miniBatch, self.defaultTrainingRate)
            self.logEpochComplete()
            self.saveNetworkState()
        self.logTrainingComplete()
    
    def getTestResult(self, testData: TestImage):
        return np.argmax(self.network.feedforward(testData.scaledData()))

    def __testNetwork(self, testSet: list[TestImage]):
        """Tests the network against the test dataset, passing a confusion matrix to the logger when done."""
        confusionMatrix = np.zeros((len(CIFAR_LABELS), len(CIFAR_LABELS)), dtype=int)
        print("Beginning testing...")

        for item in testSet:
            result = self.getTestResult(item)
            confusionMatrix[result][item.label] += 1
        
        self.logTestingComplete(confusionMatrix)
        #logger handles calculations like trace (correct) and sum (samples)
    
    def beginTesting(self, data: list[TestImage]) -> None:
        """Begins testing the network against the specified testset. 
        Only accuracy is reported, no backpropagation is performed.
        Does not affect the network; is idempotent."""
        self.__testNetwork(data)
                    
    def logPointComplete(self, caseIdx, cost, real, decision, label):
        logging.logPointComplete(self.miniBatchIdx, caseIdx, cost, real, decision, label)

    def logMinibatchComplete(self, avgCost, acc):
        logging.logMinibatchComplete(self.miniBatchIdx, self.miniBatchCount, avgCost, acc)
    
    def logEpochComplete(self):
        logging.logEpochComplete(self.currentEpoch, self.epochCount, 
            round((self.epochCostSum / self.miniBatchCount), 4), 
            self.epochCorrect, self.trainingDataSize)

    def logTrainingComplete(self):
        logging.logTrainingComplete(self.correct, self.samples)
    
    def logTestingComplete(self, confusionMatrix):
        logging.logTestingComplete(confusionMatrix)

    def saveNetworkState(self):
        NeuralNetwork4File(self.network, self.currentEpoch).logToFile()
    
class NeuralNetwork4File():
    def __init__(self, fullNetwork: NeuralNetwork4, epoch):
        self.inputNeuronCount = INPUT_LENGTH
        self.h1NeuronCount = len(fullNetwork.h1Biases)
        self.h2NeuronCount = len(fullNetwork.h2Biases)
        self.outputNeuronCount = OUTPUT_LENGTH
        self.h1Weights = fullNetwork.h1Weights
        self.h1Biases = fullNetwork.h1Biases
        self.h2Weights = fullNetwork.h2Weights
        self.h2Biases = fullNetwork.h2Biases
        self.outputWeights = fullNetwork.outputWeights
        self.outputBiases = fullNetwork.outputBiases
        self.epoch = epoch
    
    def logToFile(self):
        repickle(f"./results/{logging.F_TIMESTAMP}/trained/modelEpoch{self.epoch}", self)

def NeuralNetwork4Backtest(model_timestamp: str, testSet: list[TestImage]):
    print(f"Beginning backtest of model {model_timestamp}...")
    epochCount = len(os.listdir(f"./results/{model_timestamp}/trained/"))
    if not (logging.initBacktestLog(model_timestamp)):
        print(f"Failsafe engaged!")
        print(f"Model {model_timestamp} already has a backtest log.")
        print(f"Delete it to re-backtest.")
        return 0
    samples = len(testSet)
    for epochNum in range(1, epochCount+1):
        modelFile: NeuralNetwork4File
        modelFile = unpickle(f"./results/{model_timestamp}/trained/modelEpoch{epochNum}")
        newModel = NeuralNetwork4(modelFile.inputNeuronCount, modelFile.h1NeuronCount, 
            modelFile.h2NeuronCount, modelFile.outputNeuronCount)
        newModel.h1Weights = modelFile.h1Weights
        newModel.h1Biases = modelFile.h1Biases
        newModel.h2Weights = modelFile.h2Weights
        newModel.h2Biases = modelFile.h2Biases
        newModel.outputWeights = modelFile.outputWeights
        newModel.outputBiases = modelFile.outputBiases
        correct = 0
        for item in testSet:
            if np.argmax(newModel.feedforward(item.scaledData())) == item.label:
                correct += 1
        logging.logBacktestComplete(model_timestamp, epochNum, epochCount, correct, samples)
    print("Done backtest!")
        
        



# Main Code


CIFAR_DATA_TRAIN = np.array(unpickle("cifar10TRAIN"))
CIFAR_DATA_TEST = np.array(unpickle("cifar10TEST"))

CIFAR_DATA_TRAIN_SAMPLES = len(CIFAR_DATA_TRAIN)
CIFAR_DATA_TEST_SAMPLES = len(CIFAR_DATA_TEST)

LOGGING_LEVEL = 2

RNG_MEAN = 0
RNG_STDDEV = 1

layerH1NeuronCount = 150
layerH2NeuronCount = 40
trainingRate = 0.25
miniBatchSize = 100
epochCount = 3

logging.initEpochLogging(LOGGING_LEVEL)

cifarNetwork = NeuralNetwork4(INPUT_LENGTH, layerH1NeuronCount, layerH2NeuronCount, OUTPUT_LENGTH)
cifarNetworkTrainer = NeuralNetwork4Trainer(cifarNetwork, trainingRate)
cifarNetworkTrainer.setupTrainingData(CIFAR_DATA_TRAIN)

start_time = time.time()
cifarNetworkTrainer.beginTraining(epochCount, miniBatchSize)
cifarNetworkTrainer.beginTesting(CIFAR_DATA_TEST)

#repickle(f"./results/{logging.F_TIMESTAMP}/cifarModelTrained", cifarNetworkTrainer)
logging.logHyperparameters(
        {"mean":RNG_MEAN, 
        "stddev": RNG_STDDEV,
        "h1": layerH1NeuronCount,
        "h2": layerH2NeuronCount,
        "eta": trainingRate,
        "mbsize": miniBatchSize,
        "epochs": epochCount}
    )

# NeuralNetwork4Backtest("2023Jul23_15h45m36s", CIFAR_DATA_TEST)