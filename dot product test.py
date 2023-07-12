import numpy as np

#Which weightsN vector is closer to the input?
inputVector = np.array([1.72, 1.73])
weights1 = np.array([1.26, 0])
weights2 = np.array([2.17, 0.32])

#Find the dot product:
dotProduct1 = np.dot(inputVector, weights1)
dotProduct2 = np.dot(inputVector, weights2)

print(f"1: {dotProduct1}, 2: {dotProduct2}")