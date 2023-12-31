import numpy as np
import time
from datetime import datetime
import os
import csv
import json

F_TIMESTAMP = 0

startTime = time.time()

LoggingLevel = 0
#0 = standard
#1 = epoch level
#2 = minibatch level
#3 = training case level

def getElapsedTime():
    return round(time.time() - startTime, 3)

def initEpochLogging(level):
    global F_TIMESTAMP
    F_TIMESTAMP = datetime.now().strftime("%Y%b%d_%Hh%Mm%Ss")
    global LoggingLevel
    LoggingLevel = level
    
    os.makedirs(f"./results/{F_TIMESTAMP}/")
    os.makedirs(f"./results/{F_TIMESTAMP}/trained/")
    with open(f"./results/{F_TIMESTAMP}/trainingLog.csv", "a+", newline="\n") as csvfile:
        sopleWriter = csv.writer(csvfile, delimiter = ' ')
        sopleWriter.writerow(["Epoch", "MaxEpoch", "AvgCost", "EpochAcc", "Elapsed"])
    csvfile.close()

def logPointComplete(batchIdx, caseIdx, cost, success, decision, label):
    print(f"Training... MB: {batchIdx} \t Obj: {caseIdx} \t Cost: {cost} \t Result: {success} [{decision}R, {label}E]")

def logMinibatchComplete(batch, maxBatch, avgCost, acc):
    print(f"Minibatch {batch}/{maxBatch} complete! \t Cost: {avgCost} \t Acc: {acc}% \t Elapsed: {getElapsedTime()}s")

def logEpochComplete(epoch, maxEpoch, avgCost, epochCorrect, trainingDataSize):
    epochAcc = round(100*epochCorrect/trainingDataSize, 3)
    if LoggingLevel > 0:
        print(f"Epoch {epoch}/{maxEpoch} complete! \t Cost: {avgCost} \t Acc: {epochAcc}% \t Elapsed: {getElapsedTime()}s")
    with open(f"./results/{F_TIMESTAMP}/trainingLog.csv", "a+", newline="\n") as csvfile:
        sopleWriter = csv.writer(csvfile, delimiter = ' ')
        sopleWriter.writerow([epoch, maxEpoch, avgCost, epochAcc, getElapsedTime()])
    csvfile.close()

def logTrainingComplete(correct, samples):
    print(f"Training complete! \t Average accuracy:{round(100*correct/samples, 3)}%")

def logTestingComplete(confusionMatrix):
    correct = np.trace(confusionMatrix)
    testSamples = np.sum(confusionMatrix)
    if LoggingLevel > 0:
        print("\nTesting complete!")
        print(f"Confusion matrix: \n {confusionMatrix} \n")
        print(f"Total accuracy: {correct} samples correct out of {testSamples} samples [{correct*100/testSamples}%]")
    with open(f"./results/{F_TIMESTAMP}/TEST_LOG.txt", "w", newline="\n") as file:
        file.write("Testing complete!")
        file.write(f"Confusion matrix: \n {confusionMatrix} \n")
        file.write(f"Total accuracy: {correct} samples correct out of {testSamples} samples [{correct*100/testSamples}%]")
    np.savetxt(f"./results/{F_TIMESTAMP}/TEST_CONFUSION.csv", confusionMatrix, fmt="%d", delimiter=",")

def logHyperparameters(hpDict):
    with open(f"./results/{F_TIMESTAMP}/hyperparameters.txt", "w") as file:
        file.write(json.dumps(hpDict, indent=2))
    file.close()

def initBacktestLog(model_timestamp):
    #if os.path.isfile(f"./results/{model_timestamp}/backtestLog.csv"): return 0
    with open(f"./results/{model_timestamp}/backtestLog.csv", "w+", newline="\n") as csvfile:
        sopleWriter = csv.writer(csvfile, delimiter = ' ')
        sopleWriter.writerow(["Epoch", "MaxEpoch", "Correct", "Samples"])
    csvfile.close()
    return 1

def logBacktestComplete(model_timestamp, epoch, maxEpoch, correct, samples):
    print(f"Backtested epoch {epoch}/{maxEpoch}! \t {correct}/{samples} [{round(correct*100/samples, 2)}%]")
    with open(f"./results/{model_timestamp}/backtestLog.csv", "a+", newline="\n") as csvfile:
        sopleWriter = csv.writer(csvfile, delimiter = ' ')
        sopleWriter.writerow([epoch, maxEpoch, correct, samples])
    csvfile.close()
