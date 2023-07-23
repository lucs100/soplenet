import pickle, os, shutil, json
from SopleNet import NeuralNetwork4, NeuralNetwork4Trainer

def unpickle(filepath):
    with open(filepath, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    fo.close()
    return dict

trialList = os.listdir("./results")

def cleanInterruptedTrials():
    for trial in trialList[0:len(trialList)-2]: #protects latest, just in case
        finished = os.path.exists(f"./results/{trial}/hyperparameters")
        print(f"{trial}: \t {finished}")
        if not finished:
            shutil.rmtree(f"./results/{trial}/")

def convertPickledHyperparams():
    for trial in trialList:
        try:
            hyperparams = unpickle(f"./results/{trial}/hyperparameters")
            hyperparamsDict = {
                "mean": hyperparams[0], 
                "stddev": hyperparams[1],
                "h1": hyperparams[2],
                "h2": hyperparams[3],
                "eta": hyperparams[4],
                "mbsize": hyperparams[5],
                "epochs": hyperparams[6]
            }
            #os.remove(f"./results/{trial}/hyperparameters")
            with open(f"./results/{trial}/hyperparameters.txt", 'w') as file:
                file.write(json.dumps(hyperparamsDict, indent=2))
                file.close()
        except EOFError:
            print(f"Trial {trial} has no hyperparameters! Skipping!")


#Make sure soplenet's main is commented out!