import random


# import numpy as np


class NeuralNetwork:
    """
    Class representing a NeuralNetwork.
    Description:
        - I input nodes
        - [X_1, ..., X_N]: N hidden layers with X_i hidden nodes in layer i (1 <= i <= N)
        - O output nodes
    https://en.wikipedia.org/wiki/Artificial_neural_network
    """

    def __init__(self, I, H, O, activation):
        self.amount_input = I
        self.amount_hidden = H  # array of amount of hidden nodes per hidden layer
        self.amount_output = O
        self.activation = activation
        self.weights = [None] * (len(H) + 1)  # 1 to every hidden layer + 1 set of weights to the output layer

        # Initialize all weight matrices
        self.weights[0] = [[random.uniform(-1, 1) for _ in range(self.amount_input + 1)] for _ in
                           range(self.amount_hidden[0])]
        for i in range(1, len(H)):
            w = [[random.uniform(-1, 1) for _ in range(self.amount_hidden[i - 1] + 1)] for _ in
                 range(self.amount_hidden[i])]
            self.weights[i] = w
        self.weights[len(H)] = [[random.uniform(-1, 1) for _ in range(self.amount_hidden[len(H) - 1] + 1)] for _ in
                                range(self.amount_output)]
        # Note: we do + 1 after range because we want the bias

    def feedforward(self, inputs):
        """
        Use the feedforward algorithm to compute the output according to our weights.
        https://en.wikipedia.org/wiki/Feedforward_neural_network
        """
        # We always append "1" to input, as this is used for the calculation of the bias.
        pass  # TODO: Implement feedforward algorithm

    def backpropagate(self, inputs, output_labels):
        """
        Adapt the weights according to the labels of this sample using the backtracking algorithm.
        https://en.wikipedia.org/wiki/Backpropagation
        """
        pass  # TODO: Implement backpropagation
