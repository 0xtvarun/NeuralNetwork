import numpy as np

def sigmoid(z):
        return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

if __name__ == '__main__':
    import sys
    sys.exit(1)