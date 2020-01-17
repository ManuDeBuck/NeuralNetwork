import random


class Perceptron:
    """
    Implementation of a single Perceptron.
    """

    def __init__(self, amount_inputs, activation, learning_rate):
        self.weights = [random.uniform(-1, 1) for _ in range(amount_inputs)]
        self.weights.append(random.uniform(-1, 1)) # Add bias value in weights

        self.activation = activation
        self.learning_rate = learning_rate

    def guess(self, input):
        extended_input = input[:]
        extended_input.append(1) # input for bias
        # compute sum
        w_sum = sum([inp * w for inp, w in zip(input, self.weights)])
        # compute activation degree
        output = self.activation(w_sum)
        return output

    def learn(self, input, output):
        extended_input = input[:]
        extended_input.append(1)  # input for bias

        guess = self.guess(input)
        error = output - guess

        # Adjust all weigths
        for i, w in enumerate(self.weights):
            # Hebbian learning rule
            self.weights[i] += error * extended_input[i] * self.learning_rate
