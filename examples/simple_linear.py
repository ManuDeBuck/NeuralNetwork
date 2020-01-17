import random
from lib.perceptron import Perceptron
from lib.activation import sign_activation

A = 54.12
B = 31.2

MAX_SIZE = 1000

LEARNING_RATE = 10 ** -8


def main(amount):
    perceptron = Perceptron(2, sign_activation, LEARNING_RATE)

    samples = get_samples(amount)
    sols = [s[1] for s in samples]

    corr, it = 0, 0
    while corr != amount:
        guesses = [perceptron.guess(sample[0]) for sample in samples]
        corr = correct(sols, guesses)
        print("Iteration {}: {}%".format(it, corr / amount))
        [perceptron.learn(sample[0], sample[1]) for sample in samples]
        it += 1


def correct(solution, guesses):
    # Count amount of right guesses
    return sum([1 if s == g else 0 for s, g in zip(solution, guesses)])


def f(x):
    # Simple linear function
    return A * x + B


def solution(x, y):
    # If point (x, y) above or underneath f(x)
    return 1 if y >= f(x) else -1


def get_samples(amount):
    random_x = [random.uniform(-1, 1) * MAX_SIZE for _ in range(amount)]
    random_y = [random.uniform(-1, 1) * MAX_SIZE for _ in range(amount)]
    samples = [
        ([x, y], solution(x, y)) for x, y in zip(random_x, random_y)
    ]
    return samples


if __name__ == '__main__':
    main(5000)
