import pickle
import numpy as np
from dataclasses import dataclass

CIFAR_DATA_TEST = []
CIFAR_DATA_TRAIN = []

@dataclass
class TestImage:
    data: np.array
    label: int

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

for batch in range(1, 6):
    currentDict = unpickle(f"cifar-10-batches-py/data_batch_{batch}")
    for idx in range(0, 10000):
        CIFAR_DATA_TRAIN.append(TestImage(data=currentDict[b'data'][idx], label=currentDict[b'labels'][idx]))

currentDict = unpickle(f"cifar-10-batches-py/test_batch")
for idx in range(0, 10000):
    CIFAR_DATA_TEST.append(TestImage(data=currentDict[b'data'][idx], label=currentDict[b'labels'][idx]))

CIFAR_LABELS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]





with open("cifar10TEST", "wb") as file:
    pickle.dump(CIFAR_DATA_TEST, file=file)


with open("cifar10TRAIN", "wb") as file:
    pickle.dump(CIFAR_DATA_TRAIN, file=file)
