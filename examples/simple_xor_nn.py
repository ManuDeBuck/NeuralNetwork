import random
from lib.neuralnetwork import NeuralNetwork
from lib.activation import sign_activation

AMOUNT_SAMPLES = 1000

LEARNING_RATE = 10 ** -8


def main(amount):
    neural_network = NeuralNetwork(2, [2], 1, sign_activation)  # 1 hidden layer with 2 hidden nodes

    samples = get_samples(amount)
    solutions = extract_solutions(samples)

    corr, it = 0, 0
    while corr != amount:
        guesses = [neural_network.feedforward(sample[0]) for sample in samples]
        corr = correct(solutions, guesses)
        print("Iteration {}: {}%".format(it, corr / amount))
        [neural_network.backpropagate(sample[0], sample[1]) for sample in samples]
        it += 1


def correct(solution, guesses):
    # Count amount of right guesses
    return sum([1 if s == g else 0 for s, g in zip(solution, guesses)])


def solution(x, y):
    # If point (x, y) above or underneath f(x)
    return x ^ y


def extract_solutions(samples):
    return [s[1] for s in samples]


def get_samples(amount):
    random_x = [random.randint(0, 1) for _ in range(amount)]
    random_y = [random.randint(0, 1) for _ in range(amount)]
    samples = [
        ([x, y], solution(x, y)) for x, y in zip(random_x, random_y)
    ]
    return samples


if __name__ == '__main__':
    main(AMOUNT_SAMPLES)
