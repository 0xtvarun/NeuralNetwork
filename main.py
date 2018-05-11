import numpy as np
from sklearn.datasets import fetch_mldata
from matplotlib import pyplot as plt

class NeuralNetwork(object):

    def __init__(self, input_layer, hidden_layer, output_layer):
        np.random.seed(1)
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer

        self.B1 = np.random.randn(self.hidden_layer, 1)
        self.W1 = np.random.randn(self.input_layer, self.hidden_layer)

        self.B2 = np.random.randn(self.output_layer, 1)
        self.W2 = np.random.randn(self.hidden_layer, self.output_layer)

    def feed_forward(self, X):
        self.z2 = np.dot(X, self.W1)# output of hidden_layer (shape = (30, 30)) 
        self.a2 = self.sigmoid(self.z2) # activation of hidden layer (shape = (30, 30))

        self.z3 = np.dot(self.a2, self.W2) # output of output_layer
        self.yHat = self.sigmoid(self.z3) # activation of output_layer

        return self.yHat    

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def cost(self, X, y):
        self.yHat = self.feed_forward(X)
        J = 0.5 * sum( (y - yHat) ** 2 )
        return J

    def cost_prime(self, X, y):
        yHat = self.feed_forward(X)

        delta3 = np.multiply(-(y - yHat), self.sigmoid_prime(self.z3))
        print(delta3.shape)
        dJdW2 = np.dot(self.a2, delta3)

        delta2 = np.multiply(delta3, self.W2) * self.sigmoid_prime(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2

    def gradients(self, X, y):
        dJdW1, dJdW2 = self.cost_prime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))


if __name__ == '__main__':
    nn = NeuralNetwork(784,30,10)

    mnist = fetch_mldata('MNIST original', data_home='./data')
    indices = np.arange(len(mnist.data))

    training_count = 60000
    testing_count = 10000

    train_idx = np.arange(0, training_count)
    test_idx = np.arange(training_count + 1, training_count + testing_count)

    X_train, Y_train = mnist.data[train_idx], mnist.target[train_idx]
    X_test, Y_test = mnist.data[test_idx], mnist.target[test_idx]

    plt.subplot(221)
    plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
    plt.subplot(222)
    plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
    plt.subplot(223)
    plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
    plt.subplot(224)
    plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
    # show the plot
    plt.show()

    # print(Y_train[0].shape)

    # yHat = nn.feed_forward(X_train[0])
    # print(yHat)
    # print(nn.gradients(X_train[0], Y_train[0]))