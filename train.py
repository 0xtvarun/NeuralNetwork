import numpy as np
from network import Network
from sklearn.datasets import fetch_mldata


if __name__ == '__main__':
    mnist = fetch_mldata('MNIST original', data_home='./data')
    indices = np.arange(len(mnist.data))

    training_count = 60000
    testing_count = 10000

    train_idx = np.arange(0, training_count)
    test_idx = np.arange(training_count + 1, training_count + testing_count)

    X_train, Y_train = mnist.data[train_idx], mnist.target[train_idx]
    X_test, Y_test = mnist.data[test_idx], mnist.target[test_idx]

    nn = Network(size=[784, 30, 10])
    # nn.SGD(X_train, 30, 10, 100.0, test_data=X_test)

