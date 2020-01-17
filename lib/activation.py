import math

"""
File containing some common used activation functions.
"""


def sign_activation(x):
    """
    Sign activation function.
    https://en.wikipedia.org/wiki/Sign_function
    """
    if x >= 0:
        return 1
    else:
        return -1


def sigmoid(x):
    """
    Sigmoid activation function.
    https://en.wikipedia.org/wiki/Sigmoid_function
    """
    return 1 / 1 + math.exp(-x)


def relu(x):
    """
    ReLU activation function.
    https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    """
    return max(0, x)
