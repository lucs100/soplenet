# soplenet
A simple 4-layer feedforward neural network created for MTE 203.

Built to classify images from the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset into one of 10 categories. Uses a cross-entropy classification loss function to backpropagate error across layers, and the mini-batch stochastic gradient descent method to apply this error to the system. Logging can be done at the epoch, mini-batch, or individual test case level.
